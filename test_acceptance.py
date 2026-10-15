"""Acceptance tests T01-T09 for the branched-network demo. Run: python test_acceptance.py (or pytest)."""
import csv, re, tempfile
from pathlib import Path

import mujoco
import network as net
from medetv import ETV_COLORS, H, MIN_GAP, Z_LIMIT, EVENT_FIELDS, TELEMETRY_FIELDS, Sim

TRAVERSED = set()


def run(out, check=None, max_t=1500.0):
    sim = Sim(out)
    while not sim.done() and sim.t < max_t:
        sim.step()
        if check:
            check(sim)
    assert sim.done(), "scenario did not complete"
    return sim


def invariants(sim):
    feet = {v: set(sim.footprint(e)) for v, e in sim.etvs.items()}
    for v, e in sim.etvs.items():
        TRAVERSED.add(e.route[e.i])
        # T03: no horizontal motion while the carrier is below the transfer limit
        if sim.z(v) > Z_LIMIT:
            assert e.v == 0.0, f"{v} moving during transfer"
        # T05: every block under an ETV is reserved by that ETV
        assert all(sim.res.owner.get(b) == v for b in feet[v]), f"{v} on unreserved block"
    ids = list(feet)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            assert not feet[a] & feet[b], f"{a}/{b} share a block"
            pa, pb = sim.etvs[a].xy, sim.etvs[b].xy
            assert ((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2) ** 0.5 >= MIN_GAP, f"{a}/{b} too close"
    for z, blocks in net.ZONE_BLOCKS.items():   # at most one ETV physically inside each choke point
        assert sum(1 for v in ids if feet[v] & blocks) <= 1, f"two ETVs inside {z}"
    assert not any(e.safety for e in sim.etvs.values())


def events(out):
    with open(Path(out) / "events.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_all():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)

        # T01 fleet + network initialisation
        sim = Sim(d / "init")
        m = sim.model
        names = lambda obj, n: {mujoco.mj_id2name(m, obj, i) for i in range(n)}
        assert set(ETV_COLORS) <= names(mujoco.mjtObj.mjOBJ_BODY, m.nbody) and len(ETV_COLORS) == 6
        geoms = names(mujoco.mjtObj.mjOBJ_GEOM, m.ngeom)
        stations = [s for s in net.STATIONS if s != "DEP"]
        assert sum(s.startswith("D") for s in stations) == 5 and {f"pod:{s}" for s in stations} <= geoms
        assert len(net.ZONES) == 3 and {f"lamp:{z}" for z in net.ZONES} <= geoms
        assert all(net.EDGES[s].kind == "siding" for s in stations)      # every station is on a branch
        sim.close()

        # T06 live status consistency, sampled mid-run
        sim = Sim(d / "r0")
        while sim.t < 90:
            sim.step()
        page = sim.render_status()
        for v, e in sim.etvs.items():
            m_ = re.search(rf"^{v} \| STATE=(\S+) \| JOB=(\S+) \| X: ([\d.-]+) m \| Y: ([\d.-]+) m", page, re.M)
            assert m_ and m_[1] == e.q and float(m_[3]) == round(e.xy[0], 2) and float(m_[4]) == round(e.xy[1], 2)
        assert page.rstrip().splitlines()[-1] == (d / "r0" / "events.log").read_text().splitlines()[-1]
        sim.close()

        sim = run(d / "r1", check=invariants)
        sim.close()
        ev = events(d / "r1")
        kinds = [e["event_type"] for e in ev]
        n_jobs = len(sim.jobs)
        assert "ERROR" not in kinds

        # T02 tracking: every ETV reached a station within tolerance
        for v in ETV_COLORS:
            arr = [e for e in ev if e["vehicle_id"] == v and e["event_type"] == "ARRIVED" and e["location"].startswith("POD-")]
            assert arr, f"{v} never reached a station"
            assert all(float(re.search(r"err=([\d.]+)", e["detail"])[1]) <= 0.02 for e in arr)

        # T03 vertical transfer: one lower/raise per load and per unload
        assert kinds.count("LOWERED") == kinds.count("RAISED") == 2 * n_jobs

        # T04 payload transactions: every carrier ends in its destination pod
        for j in sim.jobs.values():
            assert j["state"] == "DELIVERED" and j["carrier"] in sim.pods[j["dst"]]
        assert kinds.count("DELIVERED") == n_jobs and all(e.payload is None for e in sim.etvs.values())

        # T05 conflict handling: choke waits happened, all three choke points used, spine used both ways
        assert any(e["event_type"] == "WAIT" and "CHOKE-OCCUPIED" in e["detail"] for e in ev)
        assert {e["location"] for e in ev if e["event_type"] == "RESERVED"} >= set(net.ZONES)
        assert {"SJB>SJT", "SJT>SJB"} <= TRAVERSED

        # T07 traceability
        with open(d / "r1" / "events.csv") as f:
            assert next(csv.reader(f)) == EVENT_FIELDS
        with open(d / "r1" / "telemetry.csv", newline="") as f:
            rows = list(csv.reader(f))
        assert rows[0] == TELEMETRY_FIELDS and len(rows) > 6 * 300

        # T09 routing: every route is a connected path; the first is the uncongested shortest path; reroutes occur
        for _, route in sim.route_log:
            assert all(net.EDGES[a].b == net.EDGES[b].a for a, b in zip(route, route[1:])), route
        assert sim.route_log[0][1][1:-1] == net.shortest_path("DEPb", "L01a")[1]
        assert "REROUTED" in kinds

        # T08 repeatability
        run(d / "r2").close()
        assert [tuple(r.values()) for r in events(d / "r2")] == [tuple(r.values()) for r in ev]


if __name__ == "__main__":
    test_all()
    print("T01-T09 PASS")
