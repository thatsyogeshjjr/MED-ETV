# MED-ETV — simulation prototype (iteration 1b: branched network)

Six overhead ETVs deliver carriers on a branched rail network in MuJoCo. A plain-text live status page runs on port 8000.
Base spec: `docs/MED_ETV_First_Iteration (1).pdf`. Iteration 1b adds what that spec lists as out of scope for
iteration 1 (branched rails, routing, rerouting), at the user's request.

    pip install -r requirements.txt
    python medetv.py --viewer            # 3D viewer + http://127.0.0.1:8000/ (auto-refresh 1 s)
    python medetv.py --viewer --speed 4  # same, 4x real time
    python medetv.py --host 0.0.0.0      # LAN access to the status page
    python medetv.py --fast              # headless, no server, prints final status
    python test_acceptance.py            # T01-T09, ~40 s

## Network (`network.py`)
One-way clockwise loop, 72 x 28 m, rail at 3 m:

    top    (east)   D01 ICU · spine junction · D02 SURGERY · L02 LAB
    east   (south)  D03 EMERGENCY · CP-3 FIRE-DOOR-E
    bottom (west)   DEPOT · spine junction · L01 PHARMACY · D04 ONCOLOGY
    west   (north)  CP-1 FIRE-DOOR-W · D05 WARD-4
    CP-2 SPINE      bidirectional single-track cross-link, bottom <-> top

- **Stations and depot:** every station is a through-siding (diverge → station → merge). Main-line traffic flows past while an ETV lowers its carrier into the pod. The depot is a 6-slot FIFO siding.
- **Jobs:** medications go from L01 PHARMACY to the wards; specimens go from the wards to L02 LAB. Jobs are released on a timed STAT/ROUTINE schedule (`SCENARIO` in `medetv.py`).

## Control (`medetv.py`)
- **Fixed-block signalling:** every edge is split into blocks of up to 2 m. An ETV reserves blocks ahead to cover its braking distance plus clearance, stops 1.5 m before any block it can't get, and releases blocks behind its tail.
- **Choke points:** entering a choke point atomically reserves the whole zone plus the first block after it. An ETV can't stall inside a choke point, and nothing can meet head-on on the spine.
- **Routing:** Dijkstra with congestion costs, replanned at each diverge (`REROUTED reason=CONGESTION`). If the destination siding is full, the ETV loops round (`REROUTED reason=SIDING-BUSY`).
- **Dispatch:** the highest-priority, oldest job goes to the eligible ETV with the lowest ETA. ETVs return to the depot when idle.
- **Safety:** a geometric min-gap guard (1.2 m) runs alongside the reservations. A watchdog flags a suspected deadlock; it detects only, recovery is out of scope.
- **Physics:** horizontal motion is kinematic (mocap) with bounded speed and acceleration. Vertical carrier transfer is an actuated MuJoCo slide joint, and horizontal motion is inhibited while the carrier is lowered.

## 3D viewer cues
- **ETV chassis:** white = travelling, amber = waiting, blue = transferring, grey = parked, red = safety hold.
- **Rail blocks:** tinted in the colour of the ETV that holds them.
- **Lamps:** choke-point lamps are red when occupied. Station lamps are blue while being serviced.
- **Carriers:** yellow = loaded. Delivered carriers stack beside each pod.

Each run writes `runs/<timestamp>/events.log`, `events.csv` and `telemetry.csv` (append-only).
