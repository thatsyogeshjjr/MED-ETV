"""MED-ETV 01 - self-propelled overhead transport vehicle. Parametric CadQuery model.

Units: mm. Frame: X = along rail, Y = across rail (door side = -Y), Z = up.
Origin: rail bottom face, rail centreline, bogie centre.

Every parameter prefixed P_ is PROVISIONAL (read off a perspective/marketing image, or assumed).
Only the "~" dimensions printed on the design board are image-derived; even those are marked provisional there.

Run:  python model.py            -> builds, validates, writes med_etv.step / med_etv.stl
      python model.py --open 90  -> same, with door opened 90 deg
"""
import sys
import itertools
import cadquery as cq

# ---------------- rail (board: ~80 wide, ~60 high, provisional) ----------------
P_RAIL_W = 80.0          # outer width            [board ~80]
P_RAIL_H = 60.0          # outer height           [board ~60]
P_RAIL_T = 3.0           # wall thickness         [assumed]
P_RAIL_LIP_T = 4.0       # bottom lip thickness   [assumed]
P_RAIL_SLOT_W = 50.0     # bottom slot width      [assumed]
P_RAIL_LEN = 600.0       # modelled rail segment  [display only]
P_STRIP_T = 6.0          # central drive strip thickness [assumed; teeth not modelled]
P_STRIP_DEPTH = 20.0     # strip hangs this far below rail roof [assumed]

# ---------------- bogie (board: ~220 long) ----------------
P_BOGIE_L = 220.0        # chassis plate length   [board ~220]
P_BOGIE_W = 90.0         # chassis plate width    [assumed]
P_CHASSIS_T = 6.0        # chassis plate thickness
P_CHASSIS_GAP = 24.0     # rail bottom -> chassis plate top
P_SPINE_W = 44.0         # spine/drive block width (passes through slot)
P_SPINE_TOP_CLR = 5.0    # spine top -> strip bottom clearance
P_SPINE_X = (40.0, 105.0)  # spine block x-extent (each side, mirrored)
P_DRIVE_BLOCK_L = 50.0   # central drive block length

P_LOAD_D, P_LOAD_W = 22.0, 8.0     # load rollers (polyurethane) [assumed]
P_LOAD_PITCH = 120.0               # load roller wheelbase
P_AXLE_D = 8.0
P_GUIDE_D, P_GUIDE_H = 12.0, 8.0   # lateral guide rollers (vertical axis)
P_GUIDE_X = 95.0                   # guide roller x position (+/-)
P_GUIDE_Z = 18.0                   # guide roller centre height
P_DROP_D, P_DROP_W = 13.0, 8.0     # anti-drop capture rollers under lips
P_DROP_GAP = 1.0                   # capture roller -> lip underside running gap
P_DROP_BRKT_T = 5.0

P_TRACTION_D, P_TRACTION_H = 22.0, 10.0  # driven wheel + spring idler (pinch drive)
P_SHAFT_D = 8.0
P_SHAFT_CLR = 1.0                  # radial clearance of shaft bores
P_IDLER_PIN_D = 6.0
P_IDLER_TRAVEL = 4.0               # preload slot travel (spring/screw itself not modelled)

P_MOTOR_D = 37.0                   # common 37 mm gear motor [assumed]
P_GEARBOX_L, P_MOTOR_L = 26.0, 57.0
P_ENCODER_D, P_ENCODER_L = 30.0, 15.0

P_MOUNT_HOLE_D = 5.5               # M5 clearance, bogie <-> pod
P_MOUNT_HOLES = [(sx * 80.0, sy * 30.0) for sx in (-1, 1) for sy in (-1, 1)]
P_MOTOR_HOLE_D = 40.0

# ---------------- pod (board: ~200 W, ~250 H; length conflicting 220/320/350) ----------------
P_POD_L = 320.0          # [side view ~320; bottom view says ~350 -> CONFIRM]
P_POD_W = 200.0          # [board ~200]
P_POD_H = 250.0          # [board ~250]
P_POD_T = 3.0            # shell wall [assumed]
P_POD_R_V = 40.0         # vertical-edge radius [estimated]
P_POD_R_H = 25.0         # top/bottom-edge radius [estimated]
P_DOUBLER_T = 3.0        # internal reinforcement plate under pod roof

P_DOOR_W, P_DOOR_H = 220.0, 170.0  # opening in -Y face [assumed, kept inside flat region]
P_DOOR_Z0 = 35.0                   # opening bottom above pod bottom (sill retains tote)
P_DOOR_GAP = 1.0
P_LATCH_D = 12.0
P_LATCH_PROUD = 3.0
P_DOOR_OPEN_DEG = 0.0              # 0 = closed, 90 = flat open (hinged at bottom edge)

P_TOTE_L, P_TOTE_W, P_TOTE_H = 200.0, 150.0, 110.0  # [assumed]
P_TOTE_T = 2.0
P_TOTE_LIP, P_TOTE_LIP_T = 4.0, 3.0

# ---------------- derived ----------------
STRIP_Z0 = P_RAIL_H - P_RAIL_T - P_STRIP_DEPTH
SPINE_TOP = STRIP_Z0 - P_SPINE_TOP_CLR
CHASSIS_TOP = -P_CHASSIS_GAP
POD_TOP = CHASSIS_TOP - P_CHASSIS_T
POD_BOT = POD_TOP - P_POD_H
RAIL_IN = P_RAIL_W / 2 - P_RAIL_T                # inner wall |y|
LIP_MID = (P_RAIL_SLOT_W / 2 + RAIL_IN) / 2      # roller track centre |y|
DRIVE_Y = P_STRIP_T / 2 + P_TRACTION_D / 2       # wheel centre |y| (tangent to strip)
WHEEL_Z0 = STRIP_Z0 + (P_STRIP_DEPTH - P_TRACTION_H) / 2


def box(x0, x1, y0, y1, z0, z1):
    return cq.Workplane().box(x1 - x0, y1 - y0, z1 - z0).translate(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


def cyl(d, h, pnt, axis):
    return cq.Workplane().add(cq.Solid.makeCylinder(d / 2, h, cq.Vector(*pnt), cq.Vector(*axis)))


def rounded_box(l, w, h, rv, rh):
    return cq.Workplane().box(l, w, h).edges("|Z").fillet(rv).faces(">Z or <Z").edges().fillet(rh)


def rail():
    w, h, t, s = P_RAIL_W / 2, P_RAIL_H, P_RAIL_T, P_RAIL_SLOT_W / 2
    pts = [(-w, 0), (-s, 0), (-s, P_RAIL_LIP_T), (-RAIL_IN, P_RAIL_LIP_T), (-RAIL_IN, h - t),
           (RAIL_IN, h - t), (RAIL_IN, P_RAIL_LIP_T), (s, P_RAIL_LIP_T), (s, 0), (w, 0), (w, h), (-w, h)]
    return cq.Workplane("YZ").polyline(pts).close().extrude(P_RAIL_LEN).translate((-P_RAIL_LEN / 2, 0, 0))


def drive_strip():
    L = P_RAIL_LEN / 2
    return box(-L, L, -P_STRIP_T / 2, P_STRIP_T / 2, STRIP_Z0, P_RAIL_H - P_RAIL_T)


def frame():
    """Chassis plate + two roller spines + central drive block (one machined/welded frame)."""
    sw = P_SPINE_W / 2
    f = box(-P_BOGIE_L / 2, P_BOGIE_L / 2, -P_BOGIE_W / 2, P_BOGIE_W / 2, POD_TOP, CHASSIS_TOP)
    for sx in (-1, 1):
        x0, x1 = sorted((sx * P_SPINE_X[0], sx * P_SPINE_X[1]))
        f = f.union(box(x0, x1, -sw, sw, CHASSIS_TOP, SPINE_TOP))
    f = f.union(box(-P_DRIVE_BLOCK_L / 2, P_DRIVE_BLOCK_L / 2, -sw, sw, CHASSIS_TOP, SPINE_TOP))
    f = f.cut(cyl(P_SHAFT_D + 2 * P_SHAFT_CLR, SPINE_TOP - POD_TOP, (0, DRIVE_Y, POD_TOP), (0, 0, 1)))
    # idler preload slot, elongated in -Y (away from strip)
    pin = P_IDLER_PIN_D + 1
    f = f.cut(box(-pin / 2, pin / 2, -DRIVE_Y - pin / 2 - P_IDLER_TRAVEL, -DRIVE_Y + pin / 2,
                  SPINE_TOP - 15, SPINE_TOP))
    for x, y in P_MOUNT_HOLES:
        f = f.cut(cyl(P_MOUNT_HOLE_D, P_CHASSIS_T, (x, y, POD_TOP), (0, 0, 1)))
    return f


def load_roller(x, sy):
    zc = P_RAIL_LIP_T + P_LOAD_D / 2
    y_in = LIP_MID - P_LOAD_W / 2
    r = cyl(P_LOAD_D, P_LOAD_W, (x, sy * y_in, zc), (0, sy, 0))
    return r.union(cyl(P_AXLE_D, y_in - P_SPINE_W / 2, (x, sy * P_SPINE_W / 2, zc), (0, sy, 0)))


def guide_roller(x, sy):
    yc = RAIL_IN - P_GUIDE_D / 2                     # tangent to inner wall
    z1 = P_GUIDE_Z + P_GUIDE_H / 2
    r = cyl(P_GUIDE_D, P_GUIDE_H, (x, sy * yc, z1 - P_GUIDE_H), (0, 0, 1))
    r = r.union(cyl(4, 2, (x, sy * yc, z1), (0, 0, 1)))
    y0, y1 = sorted((sy * P_SPINE_W / 2, sy * (RAIL_IN - 2)))
    return r.union(box(x - 5, x + 5, y0, y1, z1 + 2, z1 + 6))


def drop_roller(x, sy):
    zc = -P_DROP_GAP - P_DROP_D / 2
    y_in = LIP_MID - P_DROP_W / 2
    yb = P_RAIL_W / 2 - P_RAIL_T                     # bracket inner face, outside lip roller
    r = cyl(P_DROP_D, P_DROP_W, (x, sy * y_in, zc), (0, sy, 0))
    r = r.union(cyl(6, yb - (y_in + P_DROP_W), (x, sy * (y_in + P_DROP_W), zc), (0, sy, 0)))
    y0, y1 = sorted((sy * yb, sy * (yb + P_DROP_BRKT_T)))
    return r.union(box(x - 6, x + 6, y0, y1, CHASSIS_TOP, -2))


def traction_wheel():
    w = cyl(P_TRACTION_D, P_TRACTION_H, (0, DRIVE_Y, WHEEL_Z0), (0, 0, 1))
    return w.union(cyl(P_SHAFT_D, WHEEL_Z0 - POD_TOP, (0, DRIVE_Y, POD_TOP), (0, 0, 1)))


def idler_wheel():
    w = cyl(P_TRACTION_D, P_TRACTION_H, (0, -DRIVE_Y, WHEEL_Z0), (0, 0, 1))
    return w.union(cyl(P_IDLER_PIN_D, WHEEL_Z0 - (SPINE_TOP - 15), (0, -DRIVE_Y, SPINE_TOP - 15), (0, 0, 1)))


def gear_motor():
    z = POD_TOP - P_GEARBOX_L - P_MOTOR_L
    m = cyl(P_MOTOR_D, P_GEARBOX_L + P_MOTOR_L, (0, DRIVE_Y, z), (0, 0, 1))
    return m.union(cyl(P_ENCODER_D, P_ENCODER_L, (0, DRIVE_Y, z - P_ENCODER_L), (0, 0, 1)))


def door_opening():
    z0 = POD_BOT + P_DOOR_Z0
    return box(-P_DOOR_W / 2, P_DOOR_W / 2, -P_POD_W / 2 - 1, -P_POD_W / 2 + P_POD_T + 1, z0, z0 + P_DOOR_H)


def pod():
    t = P_POD_T
    zc = (POD_TOP + POD_BOT) / 2
    shell = rounded_box(P_POD_L, P_POD_W, P_POD_H, P_POD_R_V, P_POD_R_H).cut(
        rounded_box(P_POD_L - 2 * t, P_POD_W - 2 * t, P_POD_H - 2 * t, P_POD_R_V - t, P_POD_R_H - t))
    shell = shell.translate((0, 0, zc)).cut(door_opening())
    for x, y in P_MOUNT_HOLES:
        shell = shell.cut(cyl(P_MOUNT_HOLE_D, t, (x, y, POD_TOP - t), (0, 0, 1)))
    return shell.cut(cyl(P_MOTOR_HOLE_D, t, (0, DRIVE_Y, POD_TOP - t), (0, 0, 1)))


def doubler():
    z1 = POD_TOP - P_POD_T
    d = box(-100, 100, -45, 45, z1 - P_DOUBLER_T, z1)
    for x, y in P_MOUNT_HOLES:
        d = d.cut(cyl(P_MOUNT_HOLE_D, P_DOUBLER_T, (x, y, z1 - P_DOUBLER_T), (0, 0, 1)))
    return d.cut(cyl(P_MOTOR_HOLE_D, P_DOUBLER_T, (0, DRIVE_Y, z1 - P_DOUBLER_T), (0, 0, 1)))


def door(open_deg=P_DOOR_OPEN_DEG):
    """Logo panel: hinged at its bottom edge (outer face), quarter-turn latch near top."""
    g, y0 = P_DOOR_GAP, -P_POD_W / 2
    z0 = POD_BOT + P_DOOR_Z0
    z1 = z0 + P_DOOR_H
    d = box(-P_DOOR_W / 2 + g, P_DOOR_W / 2 - g, y0, y0 + P_POD_T, z0 + g, z1 - g)
    zl = z1 - 15
    d = d.union(cyl(P_LATCH_D, P_POD_T + P_LATCH_PROUD, (0, y0 - P_LATCH_PROUD, zl), (0, 1, 0)))
    d = d.union(box(-4, 4, y0 + P_POD_T, y0 + P_POD_T + 2, zl, z1 + 8))   # cam tab behind wall
    return d.rotate((0, y0, z0), (1, y0, z0), open_deg)


def tote():
    z0 = POD_BOT + P_POD_T
    t = cq.Workplane().box(P_TOTE_L, P_TOTE_W, P_TOTE_H).faces(">Z").shell(-P_TOTE_T)
    t = t.translate((0, 0, z0 + P_TOTE_H / 2))
    lip = (box(-P_TOTE_L / 2 - P_TOTE_LIP, P_TOTE_L / 2 + P_TOTE_LIP, -P_TOTE_W / 2 - P_TOTE_LIP,
               P_TOTE_W / 2 + P_TOTE_LIP, z0 + P_TOTE_H - P_TOTE_LIP_T, z0 + P_TOTE_H)
           .cut(box(-P_TOTE_L / 2 + P_TOTE_T, P_TOTE_L / 2 - P_TOTE_T, -P_TOTE_W / 2 + P_TOTE_T,
                    P_TOTE_W / 2 - P_TOTE_T, z0, z0 + P_TOTE_H)))
    return t.union(lip)


def parts(open_deg=P_DOOR_OPEN_DEG):
    p = {"rail": (rail(), (0.6, 0.6, 0.62)), "drive_strip": (drive_strip(), (0.1, 0.1, 0.1)),
         "bogie_frame": (frame(), (0.78, 0.78, 0.8)), "traction_wheel": (traction_wheel(), (0.1, 0.1, 0.1)),
         "idler_wheel": (idler_wheel(), (0.1, 0.1, 0.1)), "gear_motor": (gear_motor(), (0.6, 0.6, 0.62)),
         "pod_shell": (pod(), (0.95, 0.95, 0.95)), "pod_doubler": (doubler(), (0.78, 0.78, 0.8)),
         "door_panel": (door(open_deg), (0.95, 0.95, 0.95)), "tote": (tote(), (0.7, 0.8, 0.9))}
    for sx in (-1, 1):
        for sy in (-1, 1):
            k = f"{'R' if sx > 0 else 'F'}{'L' if sy > 0 else 'D'}"   # front/rear, left/door side
            p[f"load_roller_{k}"] = (load_roller(sx * P_LOAD_PITCH / 2, sy), (0.3, 0.45, 0.65))
            p[f"guide_roller_{k}"] = (guide_roller(sx * P_GUIDE_X, sy), (0.3, 0.45, 0.65))
            p[f"antidrop_roller_{k}"] = (drop_roller(sx * P_LOAD_PITCH / 2, sy), (0.3, 0.45, 0.65))
    return p


def validate(p, tol=1e-3):
    errors = []
    for name, (wp, _) in p.items():
        s = wp.val()
        if not s.isValid():
            errors.append(f"invalid solid: {name}")
        if len(wp.solids().vals()) != 1:
            errors.append(f"{name}: {len(wp.solids().vals())} solids (expected 1)")
        if s.Volume() <= 0:
            errors.append(f"{name}: non-positive volume")
    for (a, (wa, _)), (b, (wb, _)) in itertools.combinations(p.items(), 2):
        ba, bb = wa.val().BoundingBox(), wb.val().BoundingBox()
        if ba.xmin > bb.xmax or bb.xmin > ba.xmax or ba.ymin > bb.ymax or bb.ymin > ba.ymax \
                or ba.zmin > bb.zmax or bb.zmin > ba.zmax:
            continue
        v = wa.val().intersect(wb.val()).Volume()
        if v > tol:
            errors.append(f"interference {a} x {b}: {v:.2f} mm3")
    # simple geometric sanity checks
    if gear_motor().val().BoundingBox().zmin <= POD_BOT + P_POD_T + P_TOTE_H:
        errors.append("motor reaches into tote envelope")
    if P_SPINE_W >= P_RAIL_SLOT_W:
        errors.append("spine wider than rail slot")
    if P_DOOR_W / 2 > P_POD_L / 2 - P_POD_R_V or P_DOOR_Z0 < P_POD_R_H \
            or P_DOOR_Z0 + P_DOOR_H > P_POD_H - P_POD_R_H:
        errors.append("door opening runs into pod fillets")
    if P_TOTE_L + 2 * P_TOTE_LIP > P_DOOR_W - 2 * P_DOOR_GAP:
        errors.append("tote does not pass through door opening")
    return errors


def export(p, stem="med_etv"):
    asm = cq.Assembly(name="MED_ETV_01")
    for name, (wp, color) in p.items():
        asm.add(wp, name=name, color=cq.Color(*color))
    asm.export(f"{stem}.step")
    cq.exporters.export(cq.Workplane().add(cq.Compound.makeCompound([wp.val() for wp, _ in p.values()])),
                        f"{stem}.stl", tolerance=0.1, angularTolerance=0.2)


if __name__ == "__main__":
    deg = float(sys.argv[sys.argv.index("--open") + 1]) if "--open" in sys.argv else P_DOOR_OPEN_DEG
    p = parts(deg)
    errs = validate(p)
    print("\n".join(errs) if errs else f"OK: {len(p)} valid solids, no interference")
    export(p)
    print("wrote med_etv.step, med_etv.stl")
    sys.exit(1 if errs else 0)
