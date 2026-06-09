import numpy as np

d = np.load("outputs/mpc_trace.npz")
wall, q_meas, q_ref, u, err = d["wall"], d["q_meas"], d["q_ref"], d["u"], d["err"]
n = len(wall)
print("samples:", n, " duration: %.2f s" % wall[-1])

# 1) feedback rate test
dq = np.linalg.norm(np.diff(q_meas, axis=0), axis=1)
unchanged = np.sum(dq < 1e-9)
changes = np.where(dq > 1e-9)[0]
print("\n[FEEDBACK] consecutive q_meas identical: %d / %d (%.1f%%)" % (
    unchanged, n-1, 100*unchanged/(n-1)))
print("[FEEDBACK] effective feedback rate ~ %.1f Hz" % (len(changes)/wall[-1]))
if len(changes) > 1:
    gaps = np.diff(changes)
    print("[FEEDBACK] cycles between feedback updates: mean=%.1f max=%d" % (gaps.mean(), gaps.max()))

# 2) loop timing
dt = np.diff(wall)
print("\n[LOOP] dt ms: mean=%.2f median=%.2f max=%.2f" % (1e3*dt.mean(), 1e3*np.median(dt), 1e3*dt.max()))
print("[LOOP] frac cycles faster than 8ms (catch-up bursts): %.1f%%" % (100*np.mean(dt < 0.008)))
print("[LOOP] frac cycles slower than 20ms (stalls): %.1f%%" % (100*np.mean(dt > 0.020)))

# 3) command oscillation
du = np.abs(np.diff(u, axis=0))
print("\n[CMD] |u| mean=%.4f max=%.4f   |du/cycle| mean=%.4f max=%.4f" % (
    np.abs(u).mean(), np.abs(u).max(), du.mean(), du.max()))

# 4) per joint
print("\n[PER-JOINT] flips / rms_u / max|err|deg / max|du|")
for j in range(6):
    flips = np.sum(np.abs(np.diff(np.sign(u[:,j]))) > 1)
    print("  j%d: flips=%3d  rms_u=%.4f  max|err|=%6.2f  max|du|=%.4f" % (
        j, flips, np.sqrt(np.mean(u[:,j]**2)), np.rad2deg(np.max(np.abs(err[:,j]))),
        np.max(np.abs(np.diff(u[:,j])))))

# 5) correlation: does u reverse coincide with feedback updates? (residual sawtooth check)
# look at u change magnitude on cycles where feedback updated vs not
upd = np.zeros(n-1, dtype=bool); upd[changes] = True
du_norm = np.linalg.norm(np.diff(u, axis=0), axis=1)
print("\n[SAWTOOTH] mean |du| on feedback-update cycles: %.4f" % du_norm[upd].mean())
print("[SAWTOOTH] mean |du| on stale cycles:          %.4f" % du_norm[~upd].mean())
