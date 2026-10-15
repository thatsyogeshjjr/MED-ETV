"""MED-ETV rail network: topology data, fixed blocks, routing, and the MuJoCo scene.

Layout (plan view, metres): a one-way clockwise loop around a 72 x 28 floor.
  top    y=28  eastbound   D01 ICU, spine junction, D02 SURGERY, L02 LAB
  east   x=72  southbound  D03 EMERGENCY, CP-3 FIRE-DOOR-E
  bottom y=0   westbound   DEPOT, spine junction, L01 PHARMACY, D04 ONCOLOGY
  west   x=0   northbound  CP-1 FIRE-DOOR-W, D05 WARD-4
  CP-2 SPINE: bidirectional single-track cross-link at x=30 joining bottom and top.
Every station sits on a through-siding (diverge -> station -> merge) offset 2 m inside the loop.
"""
import heapq, math

RAIL_Z = 3.0             # rail height (m)
DEPTH = 1.5              # carrier drop from rail to station pod
BLOCK_MAX = 2.0          # max fixed-block length
ENTRY_CLEAR = 4.0        # a siding is entered only if this much of it can be reserved at once
SIDING_OFFSET, DIAG = 2.0, 1.5

NODES = {
    "D01a": (4, 28), "D01b": (14, 28), "SPs": (28.5, 28), "SPn": (31.5, 28),
    "D02a": (40, 28), "D02b": (50, 28), "L02a": (54, 28), "L02b": (64, 28),
    "D03a": (72, 25), "D03b": (72, 15), "CP3a": (72, 12), "CP3b": (72, 5),
    "DEPa": (69, 0), "DEPb": (39, 0), "BSn": (31.5, 0), "BSs": (28.5, 0),
    "L01a": (26, 0), "L01b": (16, 0), "D04a": (13, 0), "D04b": (3, 0),
    "CP1a": (0, 3.5), "CP1b": (0, 9.5), "D05a": (0, 13), "D05b": (0, 23),
    "SJT": (30, 25), "SJB": (30, 3),
}
# Main loop in travel order; tuples are chamfered corner points inside an edge.
LOOP = ["D01a", "D01b", "SPs", "SPn", "D02a", "D02b", "L02a", "L02b", (70.5, 28), (72, 26.5),
        "D03a", "D03b", "CP3a", "CP3b", (72, 1.5), (70.5, 0), "DEPa", "DEPb", "BSn", "BSs",
        "L01a", "L01b", "D04a", "D04b", (1.5, 0), (0, 1.5), "CP1a", "CP1b", "D05a", "D05b", (0, 26.5), (1.5, 28)]
STATIONS = {  # id: (name, diverge node, merge node)
    "L01": ("PHARMACY", "L01a", "L01b"), "L02": ("LAB", "L02a", "L02b"),
    "D01": ("ICU", "D01a", "D01b"), "D02": ("SURGERY", "D02a", "D02b"),
    "D03": ("EMERGENCY", "D03a", "D03b"), "D04": ("ONCOLOGY", "D04a", "D04b"),
    "D05": ("WARD-4", "D05a", "D05b"), "DEP": ("DEPOT", "DEPa", "DEPb"),
}
ZONES = {"CP-1": "FIRE-DOOR-W", "CP-2": "SPINE", "CP-3": "FIRE-DOOR-E"}
STATION_RGBA = {"L01": "0.30 0.70 0.45 0.35", "L02": "0.60 0.45 0.80 0.35", "DEP": "0.5 0.5 0.5 0.25"}


def _sub(p, q): return (p[0] - q[0], p[1] - q[1])
def _len(v): return math.hypot(*v)


class Edge:
    def __init__(self, eid, a, b, pts, kind="main", zone=None, stop=None, prefix=None):
        self.id, self.a, self.b, self.pts, self.kind, self.zone, self.stop = eid, a, b, pts, kind, zone, stop
        self.cum = [0.0]
        for p, q in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + _len(_sub(q, p)))
        self.length = self.cum[-1]
        cuts = [0.0, stop - 1.0, stop + 1.0, self.length] if stop is not None else [0.0, self.length]
        self.blocks = []
        for c0, c1 in zip(cuts, cuts[1:]):
            n = max(1, math.ceil((c1 - c0) / BLOCK_MAX - 1e-9))
            for j in range(n):
                self.blocks.append((f"{prefix or eid}.{len(self.blocks)}", c0 + (c1 - c0) * j / n, c0 + (c1 - c0) * (j + 1) / n))

    def reversed(self, eid):
        """Same track driven the other way: shares block ids."""
        r = Edge(eid, self.b, self.a, self.pts[::-1], self.kind, self.zone)
        r.blocks = [(bid, self.length - s1, self.length - s0) for bid, s0, s1 in reversed(self.blocks)]
        return r

    def point(self, s):
        """(x, y, yaw) at arc length s."""
        s = min(max(s, 0.0), self.length)
        k = max(0, min(len(self.pts) - 2, next((j for j in range(len(self.cum) - 1) if s <= self.cum[j + 1]), len(self.pts) - 2)))
        p, q = self.pts[k], self.pts[k + 1]
        seg = self.cum[k + 1] - self.cum[k]
        f = (s - self.cum[k]) / seg if seg else 0.0
        return p[0] + f * (q[0] - p[0]), p[1] + f * (q[1] - p[1]), math.atan2(q[1] - p[1], q[0] - p[0])

    def sub(self, s0, s1):
        """Polyline points between arc lengths s0 and s1."""
        inner = [self.pts[j] for j in range(1, len(self.pts) - 1) if s0 < self.cum[j] < s1]
        return [self.point(s0)[:2]] + inner + [self.point(s1)[:2]]


EDGES = {}


def _add(e):
    EDGES[e.id] = e
    return e


def _build():
    names = [k for k, x in enumerate(LOOP) if isinstance(x, str)]
    for j, k in enumerate(names):
        k2 = names[(j + 1) % len(names)]
        seq = LOOP[k:k2 + 1] if k2 > k else LOOP[k:] + LOOP[:k2 + 1]
        a, b = LOOP[k], LOOP[k2]
        zone = {"CP1a": "CP-1", "CP3a": "CP-3"}.get(a)
        _add(Edge(f"{a}>{b}", a, b, [NODES[x] if isinstance(x, str) else x for x in seq], "main", zone))
    for st, (_, a, b) in STATIONS.items():
        A, B = NODES[a], NODES[b]
        d = _sub(B, A)
        u = (d[0] / _len(d), d[1] / _len(d))
        n = (u[1], -u[0])                     # inward normal of a clockwise loop
        off = lambda p, along: (p[0] + along * u[0] + SIDING_OFFSET * n[0], p[1] + along * u[1] + SIDING_OFFSET * n[1])
        pts = [A, off(A, DIAG), off(B, -DIAG), B]
        diag = math.hypot(DIAG, SIDING_OFFSET)
        length = 2 * diag + _len(d) - 2 * DIAG
        _add(Edge(st, a, b, pts, "siding", stop=length - diag - 1.5))
    for a, b in (("SPs", "SJT"), ("SJT", "SPn"), ("BSn", "SJB"), ("SJB", "BSs")):
        _add(Edge(f"{a}>{b}", a, b, [NODES[a], NODES[b]], "main", "CP-2"))
    trunk = _add(Edge("SJT>SJB", "SJT", "SJB", [NODES["SJT"], NODES["SJB"]], "main", "CP-2", prefix="SPINE"))
    _add(trunk.reversed("SJB>SJT"))


_build()
BLOCKS = {}                                   # block id -> (edge id, s0, s1), first registering edge
for _e in EDGES.values():
    for _bid, _s0, _s1 in _e.blocks:
        BLOCKS.setdefault(_bid, (_e.id, _s0, _s1))
BLOCK_ZONE = {bid: EDGES[eid].zone for bid, (eid, _, _) in BLOCKS.items() if EDGES[eid].zone}
ZONE_BLOCKS = {z: {b for b, zz in BLOCK_ZONE.items() if zz == z} for z in ZONES}
OUT = {n: [e.id for e in EDGES.values() if e.a == n and e.kind == "main"] for n in NODES}


def shortest_path(src, dst, cost=lambda e: e.length):
    """Dijkstra over main-line edges (sidings excluded). Returns (cost, [edge ids]) or (inf, None)."""
    dist, prev, pq = {src: 0.0}, {}, [(0.0, src)]
    while pq:
        d, n = heapq.heappop(pq)
        if n == dst:
            break
        if d > dist[n]:
            continue
        for eid in OUT[n]:
            e = EDGES[eid]
            nd = d + cost(e)
            if nd < dist.get(e.b, math.inf):
                dist[e.b], prev[e.b] = nd, eid
                heapq.heappush(pq, (nd, e.b))
    if dst not in dist:
        return math.inf, None
    path, n = [], dst
    while n != src:
        path.append(prev[n])
        n = EDGES[prev[n]].a
    return dist[dst], path[::-1]


def station_point(st):
    return EDGES[st].point(EDGES[st].stop)


# ---------------------------------------------------------------- MuJoCo scene
POD_TOP = RAIL_Z - 0.85 - DEPTH               # carrier bottom when fully lowered


def _beam(name, p, q, z, half_w, half_h, rgba):
    dx, dy = q[0] - p[0], q[1] - p[1]
    L = math.hypot(dx, dy)
    return (f'<geom name="{name}" type="box" pos="{(p[0] + q[0]) / 2:.3f} {(p[1] + q[1]) / 2:.3f} {z:.3f}" '
            f'euler="0 0 {math.atan2(dy, dx):.5f}" size="{L / 2 + 0.01:.3f} {half_w} {half_h}" rgba="{rgba}"/>')


def build_mjcf(etvs):
    """etvs: {id: 'r g b'} identity colours."""
    g = []
    for e in EDGES.values():
        if e.id == "SJB>SJT":
            continue                          # twin of the spine trunk: same physical track
        base = "0.95 0.65 0.10 1" if e.zone else ("0.42 0.52 0.66 1" if e.kind == "siding" else "0.62 0.64 0.68 1")
        for bid, s0, s1 in e.blocks:
            pts = e.sub(s0, s1)
            for k, (p, q) in enumerate(zip(pts, pts[1:])):
                if _len(_sub(q, p)) > 1e-6:
                    g.append(_beam(f"blk:{bid}:{k}", p, q, RAIL_Z + 0.07, 0.1, 0.06, base))
    for n, (x, y) in NODES.items():           # ceiling hangers at junctions
        g.append(f'<geom type="capsule" fromto="{x} {y} {RAIL_Z + 0.12} {x} {y} {RAIL_Z + 1.4}" size="0.03" rgba="0.35 0.37 0.4 1"/>')
    for st, (name, a, b) in STATIONS.items():
        e = EDGES[st]
        x, y, yaw = station_point(st)
        A, B = NODES[a], NODES[b]
        u = ((B[0] - A[0]) / _len(_sub(B, A)), (B[1] - A[1]) / _len(_sub(B, A)))
        n = (u[1], -u[0])
        span = _len(_sub(B, A))
        zx, zy = (A[0] + B[0]) / 2 + n[0] * 4.2, (A[1] + B[1]) / 2 + n[1] * 4.2
        rgba = STATION_RGBA.get(st, "0.30 0.55 0.85 0.30")
        g.append(f'<geom type="box" pos="{zx} {zy} 0.006" euler="0 0 {math.atan2(u[1], u[0]):.5f}" '
                 f'size="{span / 2 - 1} 1.6 0.005" rgba="{rgba}"/>')
        if st == "DEP":
            g.append(f'<site name="{st} {name}" pos="{zx} {zy} 0.4" size="0.08" rgba="1 1 1 1"/>')
            continue
        g.append(f'<geom name="pod:{st}" type="box" pos="{x} {y} {POD_TOP / 2}" euler="0 0 {yaw:.5f}" '
                 f'size="0.45 0.4 {POD_TOP / 2}" rgba="0.82 0.84 0.86 1"/>')
        g.append(f'<geom type="box" pos="{x} {y} {POD_TOP + 0.01}" euler="0 0 {yaw:.5f}" size="0.3 0.25 0.012" rgba="0.15 0.15 0.18 1"/>')
        g.append(f'<geom name="podlamp:{st}" type="sphere" pos="{x + n[0] * 0.5} {y + n[1] * 0.5} {POD_TOP + 0.1}" size="0.09" rgba="0.3 0.8 0.4 1"/>')
        g.append(f'<site name="{st} {name}" pos="{x} {y} {POD_TOP + 0.45}" size="0.04" rgba="1 1 1 1"/>')
        for k in range(8):                    # carrier slots: waiting at sources, delivered at sinks
            sx, sy = x + n[0] * 1.2 + u[0] * (-1.4 + 0.4 * k), y + n[1] * 1.2 + u[1] * (-1.4 + 0.4 * k)
            g.append(f'<geom name="slot:{st}:{k}" type="box" pos="{sx:.3f} {sy:.3f} 0.13" size="0.15 0.12 0.12" rgba="0.95 0.78 0.2 0"/>')
    for z, (eid, ox) in {"CP-1": ("CP1a>CP1b", -1.6), "CP-2": ("SJT>SJB", -2.0), "CP-3": ("CP3a>CP3b", 1.6)}.items():
        name = ZONES[z]                       # choke-point lamp posts beside the zone midpoint
        x, y, _ = EDGES[eid].point(EDGES[eid].length / 2)
        px, py = x + ox, y
        g.append(f'<geom type="cylinder" fromto="{px} {py} 0 {px} {py} 2.2" size="0.06" rgba="0.2 0.2 0.22 1"/>')
        g.append(f'<geom name="lamp:{z}" type="sphere" pos="{px} {py} 2.35" size="0.22" rgba="0.2 0.85 0.3 1"/>')
        g.append(f'<site name="{z} {name}" pos="{px} {py} 2.75" size="0.04" rgba="1 1 1 1"/>')
    v = []
    for vid, rgb in etvs.items():
        v.append(f"""
    <body name="{vid}" mocap="true" pos="0 0 {RAIL_Z}">
      <geom type="box" pos="0 0 -0.04" size="0.3 0.07 0.04" rgba="0.25 0.25 0.28 1"/>
      <geom name="{vid}:chassis" type="box" pos="0 0 -0.24" size="0.55 0.26 0.16" rgba="0.85 0.87 0.9 1"/>
      <geom name="{vid}:plate" type="box" pos="0 0 -0.09" size="0.56 0.27 0.014" rgba="{rgb} 1"/>
      <site name="{vid}" pos="0 0 0.4" size="0.03" rgba="{rgb} 1"/>
      <site name="{vid}:hoist" pos="0 0 -0.40" size="0.01" group="4"/>
      <body name="{vid}:carrier" pos="0 0 -0.70">
        <joint name="{vid}:z" type="slide" axis="0 0 -1"/>
        <geom name="{vid}:carrier" type="box" size="0.25 0.2 0.15" rgba="0.55 0.57 0.6 1"/>
        <site name="{vid}:hook" pos="0 0 0.15" size="0.01" group="4"/>
      </body>
    </body>""")
    tendons = "".join(f'<spatial width="0.012" rgba="0.1 0.1 0.12 1"><site site="{vid}:hoist"/><site site="{vid}:hook"/></spatial>'
                      for vid in etvs)
    acts = "".join(f'<position joint="{vid}:z" kp="8000" kv="1000"/>' for vid in etvs)
    return f"""<mujoco model="med-etv-network">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/><headlight ambient="0.35 0.35 0.35"/></visual>
  <default><geom contype="0" conaffinity="0"/></default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.20 0.22 0.25" rgb2="0.24 0.26 0.29" width="512" height="512"/>
    <material name="floor" texture="grid" texrepeat="36 14" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="36 -10 25" dir="0 0.5 -1" diffuse="0.7 0.7 0.7" castshadow="true"/>
    <light pos="36 40 25" dir="0 -0.5 -1" diffuse="0.3 0.3 0.3" castshadow="false"/>
    <geom type="plane" size="40 18 0.1" pos="36 14 0" material="floor"/>
    {chr(10).join("    " + x for x in g)}
    {"".join(v)}
  </worldbody>
  <tendon>{tendons}</tendon>
  <actuator>{acts}</actuator>
</mujoco>"""
