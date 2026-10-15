"""MED-ETV iteration 1b: six overhead ETVs on a branched rail network with choke points and routing.

Single process. MuJoCo holds the scene, ETV poses (mocap) and the actuated vertical carrier transfer;
network.py holds the logical rail graph. Traffic control is fixed-block signalling with exclusive
choke-point zones; routing is congestion-aware Dijkstra with replanning at diverges.
Run:  python medetv.py [--host 0.0.0.0] [--port 8000] [--viewer] [--fast]
"""
import argparse, csv, math, threading, time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import mujoco

import network as net

VMAX, AMAX, LIFT, DWELL = 1.5, 0.75, 0.5, 3.0   # m/s, m/s^2, m/s, s
H, STOP_CLEAR = 0.6, 1.5          # ETV half-length; stop this far before an unreserved block
TOL_Z, Z_LIMIT, MIN_GAP = 0.01, 0.02, 1.2
DT_PHYS, CTRL_EVERY = 0.002, 10   # 500 Hz physics, 50 Hz control
TELEMETRY_EVERY, DISPATCH_EVERY, WATCHDOG = 1.0, 0.5, 60.0
CONGESTION_BLOCK, CONGESTION_ZONE, REROUTE_GAIN = 3.0, 8.0, 3.0   # routing cost terms (s)

ETV_COLORS = {"ETV-01": "0.00 0.45 0.70", "ETV-02": "0.90 0.62 0.00", "ETV-03": "0.00 0.62 0.45",
              "ETV-04": "0.84 0.37 0.00", "ETV-05": "0.35 0.70 0.90", "ETV-06": "0.80 0.47 0.65"}
STATE_RGBA = {"TRAVELING": (0.92, 0.93, 0.95, 1), "WAITING": (1.0, 0.72, 0.1, 1), "PARKED": (0.45, 0.47, 0.5, 1),
              "IDLE": (0.45, 0.47, 0.5, 1), "SAFETY_HOLD": (0.9, 0.1, 0.1, 1)}
TRANSFER_RGBA = (0.25, 0.5, 1.0, 1)
PRIO = {"STAT": 0, "ROUTINE": 1}

# Repeatable demo: (release s, job, priority, source, destination).
# Medications leave L01 PHARMACY for the wards; specimens go from the wards to L02 LAB.
SCENARIO = [
    (0, "J-0001", "STAT", "L01", "D03"), (0, "J-0002", "ROUTINE", "L01", "D01"),
    (5, "J-0003", "ROUTINE", "L01", "D04"), (10, "J-0004", "STAT", "D02", "L02"),
    (15, "J-0005", "ROUTINE", "L01", "D05"), (20, "J-0006", "ROUTINE", "L01", "D02"),
    (30, "J-0007", "STAT", "L01", "D01"), (35, "J-0008", "ROUTINE", "D05", "L02"),
    (45, "J-0009", "ROUTINE", "L01", "D03"), (55, "J-0010", "STAT", "D04", "L02"),
    (60, "J-0011", "ROUTINE", "L01", "D02"), (70, "J-0012", "ROUTINE", "D01", "L02"),
    (80, "J-0013", "STAT", "L01", "D05"), (90, "J-0014", "ROUTINE", "D03", "L02"),
    (100, "J-0015", "ROUTINE", "L01", "D04"), (110, "J-0016", "ROUTINE", "L01", "D01"),
]

EVENT_FIELDS = ["sim_time", "vehicle_id", "job_id", "event_type", "location", "status", "detail"]
TELEMETRY_FIELDS = ["sim_time", "vehicle_id", "x", "y", "z", "v", "state", "edge", "job_id", "payload",
                    "reservations", "safety_stop"]


def station_resources(st):
    return [f"BRANCH-{st}", f"LIFT-{st}", f"POD-{st}"]


def hms(t):
    t = int(t)
    return f"{t // 3600:02d}:{t // 60 % 60:02d}:{t % 60:02d}"


class Reservations:
    def __init__(self):
        self.owner = {}

    def try_acquire(self, who, res):
        """Atomic: all or nothing. Returns the first conflicting resource, or None on success."""
        c = next((r for r in res if self.owner.get(r, who) != who), None)
        if c is None:
            for r in res:
                self.owner[r] = who
        return c

    def release(self, who, res):
        for r in res:
            if self.owner.get(r) == who:
                del self.owner[r]

    def held(self, who):
        return sorted(r for r, o in self.owner.items() if o == who)


class ETV:
    def __init__(self, vid, route, s, target_s):
        self.id, self.route, self.i, self.s, self.target_s = vid, route, 0, s, target_s
        self.dest, self.v, self.zs = route[-1], 0.0, 0.0
        self.q, self.phase, self.job, self.payload = "PARKED", "IDLE", None, None
        self.sub, self.dwell_end, self.wait, self.safety = None, 0.0, None, False
        self.checked, self.arrived, self.xy = set(), True, (0.0, 0.0, 0.0)

    @property
    def edge(self):
        return net.EDGES[self.route[self.i]]


class Sim:
    def __init__(self, out_dir, scenario=SCENARIO):
        self.model = mujoco.MjModel.from_xml_string(net.build_mjcf(ETV_COLORS))
        self.data = mujoco.MjData(self.model)
        self.lock = threading.Lock()
        self.res = Reservations()
        m = self.model
        gid = lambda name: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        self.mocap = {v: m.body(v).mocapid[0] for v in ETV_COLORS}
        self.jz = {v: m.joint(f"{v}:z").qposadr[0] for v in ETV_COLORS}
        self.g_chassis = {v: gid(f"{v}:chassis") for v in ETV_COLORS}
        self.g_carrier = {v: gid(f"{v}:carrier") for v in ETV_COLORS}
        self.g_blocks = {}
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if name.startswith("blk:"):
                self.g_blocks.setdefault(name.split(":")[1], []).append(g)
        self.base_rgba = m.geom_rgba.copy()
        self.tint = {v: [0.35 * b + 0.65 * float(c) for b, c in zip((0.62, 0.64, 0.68), col.split())] + [1.0]
                     for v, col in ETV_COLORS.items()}             # reserved blocks take the owner's colour
        self.g_lamp = {z: gid(f"lamp:{z}") for z in net.ZONES}
        self.g_podlamp = {st: gid(f"podlamp:{st}") for st in net.STATIONS if st != "DEP"}
        self.g_slots = {st: [gid(f"slot:{st}:{k}") for k in range(8)] for st in self.g_podlamp}

        # fleet starts parked nose-to-tail in the depot, ETV-01 at the front
        dep, s = net.EDGES["DEP"], net.EDGES["DEP"].stop
        self.etvs = {}
        for vid in ETV_COLORS:
            e = self.etvs[vid] = ETV(vid, ["DEP"], s, dep.stop)
            self.res.try_acquire(vid, self.footprint(e))
            self.place(e)
            s = next(s0 for _, s0, s1 in dep.blocks if s0 <= s - H < s1) - STOP_CLEAR
        mujoco.mj_forward(m, self.data)

        self.pods = {st: [] for st in self.g_podlamp}
        self.jobs = {j: {"id": j, "release": t, "prio": p, "src": a, "dst": b, "carrier": None, "etv": None,
                         "state": "PLANNED", "t_dispatch": None, "t_deliver": None}
                     for t, j, p, a, b in scenario}
        self.pending, self.carriers = [], 0
        self.events, self.route_log = [], []
        self.next_tel, self.next_dispatch, self.last_progress, self.watchdog_fired = 0.0, 0.0, 0.0, False
        self.choke_waits, self.reroutes = 0, 0
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.log_f = open(out / "events.log", "a", encoding="utf-8")
        self.csv_f = open(out / "events.csv", "a", newline="", encoding="utf-8")
        self.tel_f = open(out / "telemetry.csv", "a", newline="", encoding="utf-8")
        self.ev_csv, self.tel_csv = csv.writer(self.csv_f), csv.writer(self.tel_f)
        if self.csv_f.tell() == 0:
            self.ev_csv.writerow(EVENT_FIELDS)
        if self.tel_f.tell() == 0:
            self.tel_csv.writerow(TELEMETRY_FIELDS)

    # ---- state accessors ----
    @property
    def t(self):
        return self.data.time

    def z(self, v): return float(self.data.qpos[self.jz[v]])

    def done(self):
        return all(j["state"] == "DELIVERED" for j in self.jobs.values()) and \
            all(e.phase == "IDLE" and e.route[e.i] == "DEP" and e.v == 0 and (e.arrived or e.q == "PARKED")
                for e in self.etvs.values())

    # ---- logging ----
    def log(self, etype, vid="", job="", location="", detail="", status="OK"):
        level = "ERROR" if etype == "ERROR" else "INFO"
        line = f"[{hms(self.t)}] {level} {vid or '-'} {etype} location={location or '-'} job={job or 'NONE'}" \
               + (f" {detail}" if detail else "")
        self.events.append(line)
        self.log_f.write(line + "\n"); self.log_f.flush()
        self.ev_csv.writerow([f"{self.t:.3f}", vid, job, etype, location, status, detail]); self.csv_f.flush()

    def telemetry(self):
        for v, e in self.etvs.items():
            x, y, _ = e.xy
            self.tel_csv.writerow([f"{self.t:.3f}", v, f"{x:.3f}", f"{y:.3f}", f"{self.z(v):.3f}", f"{e.v:.3f}",
                                   e.q, e.route[e.i], e.job["id"] if e.job else "", e.payload or "",
                                   " ".join(self.res.held(v)), int(e.safety)])
        self.tel_f.flush()

    def hold(self, e, reason, resource=""):
        """Safe wait; logged once per distinct reason."""
        e.q = "WAITING"
        if e.wait != (reason, resource):
            e.wait = (reason, resource)
            if reason == "CHOKE-OCCUPIED":
                self.choke_waits += 1
            self.log("WAIT", e.id, e.job["id"] if e.job else "", resource,
                     f"reason={reason}" + (f" resource={resource}" if resource else ""), status="HOLD")

    # ---- geometry along a route ----
    def place(self, e):
        e.xy = e.edge.point(e.s)
        k = self.mocap[e.id]
        self.data.mocap_pos[k] = (e.xy[0], e.xy[1], net.RAIL_Z)
        self.data.mocap_quat[k] = (math.cos(e.xy[2] / 2), 0, 0, math.sin(e.xy[2] / 2))

    def route_blocks(self, e):
        """Yield (route index, block, d0, d1) from the ETV's tail to its target; d is distance from the ETV."""
        start = max(0, e.i - 1)
        base = -e.s - (net.EDGES[e.route[start]].length if start < e.i else 0.0)
        for k in range(start, len(e.route)):
            ed = net.EDGES[e.route[k]]
            for bid, s0, s1 in ed.blocks:
                if k == len(e.route) - 1 and s0 >= e.target_s + H:
                    return
                if base + s1 > -H:
                    yield k, bid, base + s0, base + s1
            base += ed.length

    def footprint(self, e):
        return [bid for _, bid, d0, d1 in self.route_blocks(e) if d0 < H]

    def dist_to_target(self, e):
        return sum(net.EDGES[r].length for r in e.route[e.i:-1]) - e.s + e.target_s

    def committed(self, e):
        """Furthest route index the ETV holds blocks on: replanning starts after it."""
        return max([e.i] + [k for k, bid, _, _ in self.route_blocks(e) if self.res.owner.get(bid) == e.id])

    # ---- routing ----
    def cost_for(self, vid):
        def cost(ed):
            if ed.zone:                               # occupied choke point: one expected-wait penalty per zone edge
                return ed.length / VMAX + (CONGESTION_ZONE if self.res.owner.get(ed.zone, vid) != vid else 0.0)
            held = sum(1 for b, _, _ in ed.blocks if self.res.owner.get(b, vid) != vid)
            return ed.length / VMAX + CONGESTION_BLOCK * held
        return cost

    def nodes_of(self, route):
        return ">".join([net.EDGES[route[0]].a] + [net.EDGES[r].b for r in route])

    def set_route(self, e, route, reason=None):
        e.route, e.checked, e.arrived = route, set(), False
        if e.i > 1:                                   # keep one edge behind for the tail
            e.route, e.i = e.route[e.i - 1:], 1
        self.route_log.append((e.id, list(e.route)))
        etype = "REROUTED" if reason else "ROUTED"
        if reason:
            self.reroutes += 1
        self.log(etype, e.id, e.job["id"] if e.job else "", e.dest,
                 (f"reason={reason} " if reason else "") + f"via={self.nodes_of(e.route[e.i:])}")

    def plan(self, e, dest):
        c = self.committed(e)
        prefix, tgt = e.route[:c + 1], net.EDGES[dest]
        e.dest, e.target_s = dest, tgt.stop
        if prefix[-1] == dest and c == e.i and e.s <= tgt.stop + 1e-6:   # already on the destination siding
            return self.set_route(e, prefix)
        _, path = net.shortest_path(net.EDGES[prefix[-1]].b, tgt.a, self.cost_for(e.id))
        self.set_route(e, prefix + path + [dest])

    def eta(self, e, st):
        if e.route[e.i] == st and abs(e.s - net.EDGES[st].stop) < 1e-6:
            return 0.0
        cost, _ = net.shortest_path(net.EDGES[e.route[self.committed(e)]].b, net.EDGES[st].a, self.cost_for(e.id))
        return cost

    def consider_reroute(self, e, k):
        """At an upcoming diverge: switch to a cheaper path if congestion makes it worthwhile."""
        node = net.EDGES[e.route[k]].a
        if len(net.OUT[node]) < 2:
            return False
        cost = self.cost_for(e.id)
        cur = sum(cost(net.EDGES[r]) for r in e.route[k:-1])
        best_cost, best = net.shortest_path(node, net.EDGES[e.dest].a, cost)
        if best is None or best == e.route[k:-1] or best_cost + REROUTE_GAIN >= cur:
            return False
        self.set_route(e, e.route[:k] + best + [e.dest], "CONGESTION")
        return True

    def reroute_siding_busy(self, e, k):
        """Destination siding full: stay on the main line and come round again."""
        node = net.EDGES[e.route[k]].a
        main = net.OUT[node][0]
        _, path = net.shortest_path(net.EDGES[main].b, node, self.cost_for(e.id))
        self.set_route(e, e.route[:k] + [main] + path + [e.route[k]], "SIDING-BUSY")

    # ---- traffic control ----
    def entry_group(self, e, k, bid):
        ed = net.EDGES[e.route[k]]
        z = net.BLOCK_ZONE.get(bid)
        if z and self.res.owner.get(z) != e.id:
            # whole choke zone + first block beyond it, atomically: never stall inside, never meet head-on
            after = next((b for r in e.route[k:] for b, _, _ in net.EDGES[r].blocks if net.BLOCK_ZONE.get(b) != z), None)
            return [z] + sorted(net.ZONE_BLOCKS[z]) + ([after] if after else []), "zone"
        if ed.kind == "siding" and bid == ed.blocks[0][0]:
            return [b for b, s0, _ in ed.blocks if s0 < net.ENTRY_CLEAR], "siding"
        return [bid], "block"

    def move(self, e, dt):
        """Advance along the route under block protection. Returns True when stopped at the target."""
        if self.z(e.id) > Z_LIMIT:                    # carrier below rail-transfer limit: no horizontal motion
            e.v = 0.0
            return False
        need = e.v ** 2 / (2 * AMAX) + STOP_CLEAR + H + 0.5
        limit, blocked = self.dist_to_target(e), None
        for k, bid, d0, _ in self.route_blocks(e):
            if d0 > need:
                break
            if self.res.owner.get(bid) == e.id:
                continue
            if k > e.i and bid == net.EDGES[e.route[k]].blocks[0][0] and k not in e.checked:
                e.checked.add(k)
                if self.consider_reroute(e, k):
                    return False
            group, kind = self.entry_group(e, k, bid)
            c = self.res.try_acquire(e.id, group)
            if c:
                if kind == "siding" and k == len(e.route) - 1:
                    self.reroute_siding_busy(e, k)
                    return False
                blocked, limit = (kind, group[0] if kind == "zone" else c), min(limit, d0 - STOP_CLEAR)
                break
            if kind != "block":
                self.log("RESERVED", e.id, e.job["id"] if e.job else "", group[0] if kind == "zone" else e.route[k],
                         f"{kind} blocks={len(group) - (kind == 'zone')}")
        if limit <= 1e-6:
            e.v = 0.0
        else:
            e.v = min(e.v + AMAX * dt, VMAX, math.sqrt(2 * AMAX * limit))
            e.s += min(e.v * dt, limit)
            while e.s >= e.edge.length and e.i < len(e.route) - 1:
                e.s -= e.edge.length
                e.i += 1
            self.place(e)
        if blocked and e.v == 0.0:
            if e.phase == "IDLE" and e.route[e.i] == "DEP":
                e.q = "PARKED"
            else:
                self.hold(e, "CHOKE-OCCUPIED" if blocked[0] == "zone" else
                          "SIDING-OCCUPIED" if blocked[0] == "siding" else "SEGMENT-OCCUPIED", blocked[1])
            return False
        e.wait = None
        if blocked is None and self.dist_to_target(e) <= 1e-6:
            e.v = 0.0
            return True
        e.q = "TRAVELING"
        return False

    def release_behind(self, e):
        """Keep only the contiguous run of owned blocks from the tail forward (plus whole zones touched)."""
        keep = set()
        for _, bid, _, _ in self.route_blocks(e):
            if self.res.owner.get(bid) != e.id:
                break
            keep.add(bid)
        for z, blocks in net.ZONE_BLOCKS.items():
            if keep & blocks:
                keep |= blocks | {z}
        self.res.release(e.id, [r for r in self.res.held(e.id) if (r in net.BLOCKS or r in net.ZONES) and r not in keep])

    def guard(self):
        """Secondary geometric check, independent of the reservation table."""
        ids = list(self.etvs)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pa, pb = self.etvs[a].xy, self.etvs[b].xy
                gap = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
                if gap < MIN_GAP:
                    for v in (a, b):
                        e = self.etvs[v]
                        if not e.safety:
                            e.safety, e.q, e.v = True, "SAFETY_HOLD", 0.0
                            self.log("ERROR", v, e.job["id"] if e.job else "", e.route[e.i],
                                     f"reason=MIN-GAP gap={gap:.2f}", status="FAIL")

    # ---- jobs ----
    def release_jobs(self):
        for j in self.jobs.values():
            if j["state"] == "PLANNED" and self.t >= j["release"]:
                self.carriers += 1
                j["carrier"], j["state"] = f"Carrier-{self.carriers:03d}", "QUEUED"
                self.pods[j["src"]].append(j["carrier"])
                self.pending.append(j["id"])
                self.log("QUEUED", "", j["id"], f"POD-{j['src']}",
                         f"priority={j['prio']} route={j['src']}>{j['dst']} payload={j['carrier']}")

    def eligible(self, e):
        if e.phase != "IDLE" or e.safety:
            return False
        if "DEP" not in e.route[e.i:self.committed(e) + 1]:
            return True
        # depot is FIFO: only the ETV with nobody ahead of it may leave
        return not any(o is not e and o.route[o.i] == "DEP" and (e.route[e.i] != "DEP" or o.s > e.s)
                       for o in self.etvs.values())

    def dispatch(self):
        for jid in sorted(self.pending, key=lambda j: (PRIO[self.jobs[j]["prio"]], self.jobs[j]["release"], j)):
            j = self.jobs[jid]
            cands = [e for e in self.etvs.values() if self.eligible(e)]
            if not cands:
                return
            e = min(cands, key=lambda e: (self.eta(e, j["src"]), e.id))
            self.pending.remove(jid)
            j["state"], j["etv"], j["t_dispatch"] = "ASSIGNED", e.id, self.t
            e.job, e.phase = j, "TO_SRC"
            self.log("DISPATCHED", e.id, jid, f"POD-{j['src']}", f"priority={j['prio']} eta={self.eta(e, j['src']):.1f}s")
            self.plan(e, j["src"])

    # ---- control ----
    def step(self):
        """One control tick: jobs, dispatch, guard, per-ETV control, visuals, then CTRL_EVERY physics steps."""
        dt = DT_PHYS * CTRL_EVERY
        self.release_jobs()
        if self.t >= self.next_dispatch:
            self.dispatch()
            self.next_dispatch += DISPATCH_EVERY
        self.guard()
        for e in self.etvs.values():
            if not e.safety:
                self.control(e, dt)
            self.release_behind(e)
        for k, e in enumerate(self.etvs.values()):
            self.data.ctrl[k] = e.zs
        self.watchdog()
        self.update_visuals()
        for _ in range(CTRL_EVERY):
            mujoco.mj_step(self.model, self.data)
        if self.t >= self.next_tel:
            self.telemetry()
            self.next_tel += TELEMETRY_EVERY

    def control(self, e, dt):
        if e.phase in ("LOAD", "UNLOAD"):
            return self.transfer(e, dt)
        if not self.move(e, dt) or e.arrived:
            return
        e.arrived = True
        x, y, _ = net.station_point(e.dest)
        mp = self.data.mocap_pos[self.mocap[e.id]]
        self.log("ARRIVED", e.id, e.job["id"] if e.job else "", f"POD-{e.dest}" if e.dest != "DEP" else "DEPOT",
                 f"x={mp[0]:.3f} y={mp[1]:.3f} err={math.hypot(mp[0] - x, mp[1] - y):.3f}")
        if e.phase == "TO_SRC":
            e.phase = "LOAD"
        elif e.phase == "TO_DST":
            e.phase = "UNLOAD"
        else:
            e.q = "PARKED"

    def transfer(self, e, dt):
        j, kind = e.job, ("load" if e.phase == "LOAD" else "unload")
        st = j["src"] if kind == "load" else j["dst"]
        z = self.z(e.id)
        if e.sub is None:
            c = self.res.try_acquire(e.id, station_resources(st))
            if c:
                return self.hold(e, "STATION-OCCUPIED", c)
            e.wait, e.sub, e.q = None, "lower", "LOWERING"
            self.log("RESERVED", e.id, j["id"], f"POD-{st}", "resources=" + ",".join(station_resources(st)))
        if e.sub == "lower":
            e.zs = min(net.DEPTH, e.zs + LIFT * dt)
            if abs(z - net.DEPTH) < TOL_Z:
                self.log("LOWERED", e.id, j["id"], f"POD-{st}", f"z={z:.3f}")
                e.sub, e.dwell_end, e.q = "dwell", self.t + DWELL, "LOADING" if kind == "load" else "UNLOADING"
        elif e.sub == "dwell" and self.t >= e.dwell_end:
            if kind == "load":
                self.pods[st].remove(j["carrier"])
                e.payload, j["state"] = j["carrier"], "IN-TRANSIT"
                self.log("LOADED", e.id, j["id"], f"POD-{st}", f"payload={j['carrier']}")
            else:
                self.pods[st].append(j["carrier"])
                e.payload, j["state"], j["t_deliver"] = None, "DELIVERED", self.t
                self.log("UNLOADED", e.id, j["id"], f"POD-{st}", f"payload={j['carrier']}")
                self.log("DELIVERED", e.id, j["id"], f"POD-{st}",
                         f"payload={j['carrier']} latency={self.t - j['release']:.1f}s")
            e.sub, e.q = "raise", "RAISING"
        elif e.sub == "raise":
            e.zs = max(0.0, e.zs - LIFT * dt)
            if z < TOL_Z:
                self.log("RAISED", e.id, j["id"], f"POD-{st}", f"z={z:.3f}")
                self.res.release(e.id, station_resources(st))
                e.sub = None
                if kind == "load":
                    e.phase = "TO_DST"
                    self.plan(e, j["dst"])
                else:
                    e.phase, e.job = "IDLE", None
                    self.plan(e, "DEP")

    def watchdog(self):
        # ponytail: detects gridlock only; add recovery (back-off/reroute of a victim ETV) if networks grow
        busy = any(e.job for e in self.etvs.values())
        if not busy or any(e.v > 0 or e.sub for e in self.etvs.values()):
            self.last_progress = self.t
        elif self.t - self.last_progress > WATCHDOG and not self.watchdog_fired:
            self.watchdog_fired = True
            self.log("ERROR", "", "", "NETWORK", f"reason=DEADLOCK-SUSPECTED idle_for={WATCHDOG:.0f}s", status="FAIL")

    # ---- live 3D cues (MuJoCo viewer) ----
    def update_visuals(self):
        rgba = self.model.geom_rgba
        for bid, gids in self.g_blocks.items():
            o = self.res.owner.get(bid)
            for g in gids:
                rgba[g] = self.tint[o] if o else self.base_rgba[g]
        for v, e in self.etvs.items():
            rgba[self.g_chassis[v]] = TRANSFER_RGBA if e.phase in ("LOAD", "UNLOAD") and e.sub else STATE_RGBA.get(e.q, STATE_RGBA["IDLE"])
            rgba[self.g_carrier[v]] = (0.98, 0.78, 0.15, 1) if e.payload else (0.55, 0.57, 0.6, 1)
        for z, g in self.g_lamp.items():
            rgba[g] = (0.95, 0.15, 0.1, 1) if z in self.res.owner else (0.2, 0.85, 0.3, 1)
        for st, g in self.g_podlamp.items():
            rgba[g] = (0.25, 0.5, 1.0, 1) if f"POD-{st}" in self.res.owner else (0.3, 0.8, 0.4, 1)
            for k, sg in enumerate(self.g_slots[st]):
                rgba[sg][3] = 1.0 if k < len(self.pods[st]) else 0.0

    # ---- plain-text status (rendered from live sim state only) ----
    def render_status(self, tail=400):
        delivered = [j for j in self.jobs.values() if j["state"] == "DELIVERED"]
        lat = [j["t_deliver"] - j["release"] for j in delivered]
        out = [f"MED-ETV LIVE STATUS | SIM-TIME {hms(self.t)} | SYSTEM {'COMPLETE' if self.done() else 'RUNNING'} | "
               f"ETV={len(self.etvs)} STATIONS={len(self.pods)} CHOKE-POINTS={len(net.ZONES)}",
               f"KPI | DELIVERED {len(delivered)}/{len(self.jobs)} | MEAN LATENCY {sum(lat) / len(lat) if lat else 0:.1f} s | "
               f"MAX LATENCY {max(lat, default=0):.1f} s | CHOKE WAITS {self.choke_waits} | REROUTES {self.reroutes}",
               "", "STATIONS"]
        for st, pod in self.pods.items():
            owner = self.res.owner.get(f"POD-{st}")
            x, y, _ = net.station_point(st)
            state = self.etvs[owner].q if owner else "IDLE"
            out.append(f"POD-{st} {net.STATIONS[st][0]:<10}| STATE={state} | ETV={owner or 'NONE'} | X: {x:.2f} m | "
                       f"Y: {y:.2f} m | Z: {net.DEPTH:.2f} m | P: {','.join(pod) or 'NONE'}")
        out += ["", "CHOKE POINTS"]
        for z, name in net.ZONES.items():
            owner = self.res.owner.get(z)
            waiting = [e.id for e in self.etvs.values() if e.wait and e.wait[1] == z]
            out.append(f"{z} {name:<12}| STATE={'OCCUPIED' if owner else 'FREE'} | ETV={owner or 'NONE'} | "
                       f"QUEUE={','.join(waiting) or 'NONE'}")
        out += ["", "ETVS"]
        for v, e in self.etvs.items():
            x, y, _ = e.xy
            nxt = self.nodes_of(e.route[e.i:])[:60]
            out.append(f"{v} | STATE={e.q} | JOB={e.job['id'] if e.job else 'NONE'} | X: {x:.2f} m | Y: {y:.2f} m | "
                       f"Z: {max(0.0, self.z(v)):.2f} m | V: {e.v:.2f} m/s | DEST={e.dest} | VIA={nxt} | P: {e.payload or 'NONE'}")
        out += ["", "JOBS"]
        for j in self.jobs.values():
            lat = f"{j['t_deliver'] - j['release']:.1f} s" if j["t_deliver"] else "-"
            out.append(f"{j['id']} | {j['prio']:<7} | {j['src']}>{j['dst']} | {j['state']:<10} | ETV={j['etv'] or 'NONE'} | "
                       f"P: {j['carrier'] or 'NONE'} | LATENCY {lat}")
        out += ["", f"EVENTS ({len(self.events)} total, full history in events.log)"] + (self.events[-tail:] if tail else [])
        return "\n".join(out) + "\n"

    def close(self):
        for f in (self.log_f, self.csv_f, self.tel_f):
            f.close()


def serve(sim, host, port):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            with sim.lock:
                body = sim.render_status().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Refresh", "1")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (0.0.0.0 for LAN)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--fast", action="store_true", help="run headless as fast as possible, no server")
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo passive viewer")
    ap.add_argument("--speed", type=float, default=1.0, help="real-time factor for live runs")
    ap.add_argument("--duration", type=float, default=1800.0, help="max sim seconds")
    ap.add_argument("--out", default=None, help="run directory (default runs/<timestamp>)")
    a = ap.parse_args()

    out = a.out or f"runs/{datetime.now():%Y%m%d-%H%M%S}"
    sim = Sim(out)
    print(f"logs -> {out}")
    if not a.fast:
        serve(sim, a.host, a.port)
        print(f"status -> http://{a.host}:{a.port}/")
    viewer = None
    if a.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(sim.model, sim.data)
        with viewer.lock():
            viewer.cam.lookat[:] = (36, 12, 0)
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 66, 90, -50
            viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
    start = time.perf_counter()
    try:
        while sim.t < a.duration and (viewer is None or viewer.is_running()):
            with sim.lock:
                sim.step()
            if viewer:
                viewer.sync()
            if a.fast and sim.done():
                break
            if not a.fast:
                lag = sim.t / a.speed - (time.perf_counter() - start)
                if lag > 0:
                    time.sleep(lag)
    except KeyboardInterrupt:
        pass
    print(sim.render_status(tail=0 if not a.fast else 400))
    sim.close()


if __name__ == "__main__":
    main()
