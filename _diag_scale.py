import numpy as np

d = np.load("outputs/mpc_trace.npz")
q_meas = d["q_meas"]   # measured (modified-DH) joints, (N,6)
q_ref = d["q_ref"]     # reference the MPC compared against, (N,6)
err = d["err"]

print("samples:", len(q_meas))

# ---- joint-space amplitude comparison (range = max-min over the run) ----
print("\n[JOINT RANGE]  measured vs reference (deg)")
print(" j   meas_range  ref_range   ratio")
for j in range(6):
    rm = np.rad2deg(q_meas[:, j].max() - q_meas[:, j].min())
    rr = np.rad2deg(q_ref[:, j].max() - q_ref[:, j].min())
    ratio = rm / rr if rr > 1e-9 else float("nan")
    print("  %d   %8.2f   %8.2f   %.3f" % (j, rm, rr, ratio))

# ---- best-fit multiplicative scale per joint: q_meas ~ alpha*q_ref + b ----
print("\n[BEST-FIT SCALE]  q_meas = alpha*q_ref + b  (alpha<1 => shrunk)")
for j in range(6):
    x = q_ref[:, j] - q_ref[:, j].mean()
    y = q_meas[:, j] - q_meas[:, j].mean()
    denom = np.dot(x, x)
    alpha = np.dot(x, y) / denom if denom > 1e-12 else float("nan")
    print("  j%d: alpha=%.3f" % (j, alpha))

# ---- cross-correlation lag (is it lag, not scale?) ----
print("\n[LAG CHECK] best lag (ticks) that maximizes match per joint")
for j in range(6):
    x = q_ref[:, j] - q_ref[:, j].mean()
    y = q_meas[:, j] - q_meas[:, j].mean()
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        print("  j%d: flat" % j); continue
    best_lag, best_c = 0, -2
    for lag in range(0, 40):
        a = x[:len(x)-lag]; b = y[lag:]
        c = np.corrcoef(a, b)[0, 1]
        if c > best_c:
            best_c, best_lag = c, lag
    # amplitude ratio at that lag
    a = x[:len(x)-best_lag]; b = y[best_lag:]
    amp = np.std(b)/np.std(a)
    print("  j%d: lag=%2d ticks (%.0f ms)  corr=%.3f  amp_ratio@lag=%.3f" % (
        j, best_lag, best_lag/75.0*1000, best_c, amp))

# ---- FK to pen tip, compare XY extent ----
try:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from src.ur10e import UR10e
    robot = UR10e()

    def fk_xy(qmod):
        pts = np.zeros((len(qmod), 3))
        for i, qm in enumerate(qmod):
            qc_deg = np.rad2deg(robot.DHModifiedToClassical(qm))
            T = robot.FK(qc_deg)
            pts[i] = T[:3, 3]
        return pts

    pm = fk_xy(q_meas)
    pr = fk_xy(q_ref)
    print("\n[PEN-TIP CARTESIAN EXTENT] (m)")
    for ax, name in zip(range(3), "XYZ"):
        rm = pm[:, ax].max() - pm[:, ax].min()
        rr = pr[:, ax].max() - pr[:, ax].min()
        print("  %s: meas=%.4f  ref=%.4f  ratio=%.3f" % (name, rm, rr, rm/rr if rr > 1e-9 else float('nan')))
    # path length ratio
    Lm = np.sum(np.linalg.norm(np.diff(pm[:, :2], axis=0), axis=1))
    Lr = np.sum(np.linalg.norm(np.diff(pr[:, :2], axis=0), axis=1))
    print("  XY path length: meas=%.3f  ref=%.3f  ratio=%.3f" % (Lm, Lr, Lm/Lr if Lr > 1e-9 else float('nan')))
except Exception as e:
    print("\n[FK skipped: %s]" % e)
