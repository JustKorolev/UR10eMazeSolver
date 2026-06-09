import numpy as np

d = np.load("outputs/mpc_trace.npz")
q_ref = d["q_ref"]            # (N,6) reference the MPC actually saw
dt = 1.0 / 75.0

# per-step joint-space distance between consecutive reference points
step = np.linalg.norm(np.diff(q_ref, axis=0), axis=1)   # rad per tick
vel = step / dt                                          # implied rad/s

print("samples:", len(q_ref))
print("\n[STEP SIZE]  rad/tick")
print("  mean=%.5f  median=%.5f  std=%.5f" % (step.mean(), np.median(step), step.std()))
print("  min=%.5f  max=%.5f   max/mean=%.1fx" % (step.min(), step.max(), step.max()/ (step.mean()+1e-12)))
print("  coeff of variation (std/mean)=%.2f   <-- 0 = perfectly uniform" % (step.std()/(step.mean()+1e-12)))

print("\n[IMPLIED REF SPEED]  rad/s  (= step/dt, this is the feedforward)")
print("  mean=%.3f  max=%.3f   (VJ limit = 0.6)" % (vel.mean(), vel.max()))

# acceleration implied by spacing changes (jerk in the reference itself)
acc = np.abs(np.diff(vel)) / dt
print("\n[IMPLIED REF ACCEL]  rad/s^2  (how fast the feedrate itself changes)")
print("  mean=%.3f  max=%.3f   (AJ limit = 1.2)" % (acc.mean(), acc.max()))
print("  frac of steps exceeding AJ=1.2: %.1f%%" % (100*np.mean(acc > 1.2)))

# how many steps are ~0 (robot told to barely move) vs large
print("\n[SPACING UNIFORMITY]")
print("  steps < 25%% of mean (near-stalls): %.1f%%" % (100*np.mean(step < 0.25*step.mean())))
print("  steps > 200%% of mean (lurches):    %.1f%%" % (100*np.mean(step > 2.0*step.mean())))


def resample_uniform(q, cruise=0.12, rate=75.0):
    q_un = np.unwrap(q, axis=0)
    seg = np.linalg.norm(np.diff(q_un, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = float(s[-1])
    n = max(2, int(np.ceil(L / (cruise / rate))))
    s_new = np.linspace(0.0, L, n)
    out = np.empty((n, q.shape[1]))
    for j in range(q.shape[1]):
        out[:, j] = np.interp(s_new, s, q_un[:, j])
    return out


print("\n" + "=" * 50)
print("AFTER uniform-arc-length resampling (CRUISE=0.12):")
qr2 = resample_uniform(q_ref)
step2 = np.linalg.norm(np.diff(qr2, axis=0), axis=1)
vel2 = step2 / dt
acc2 = np.abs(np.diff(vel2)) / dt
print("  points: %d -> %d" % (len(q_ref), len(qr2)))
print("  step CoV: %.2f -> %.2f" % (step.std()/step.mean(), step2.std()/step2.mean()))
print("  ref speed max: %.3f -> %.3f  (VJ=0.6)" % (vel.max(), vel2.max()))
print("  ref accel max: %.3f -> %.3f  (AJ=1.2)" % (acc.max(), acc2.max()))
print("  %% steps over AJ: %.1f%% -> %.1f%%" % (
    100*np.mean(acc > 1.2), 100*np.mean(acc2 > 1.2)))
