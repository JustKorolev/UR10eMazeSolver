import numpy as np

d = np.load("outputs/mpc_trace.npz")
q_meas = d["q_meas"]
q_ref = d["q_ref"]
u = d["u"]
err = d["err"]
wall = d["wall"]

dt_model = 1.0 / 75.0
T_real = np.median(np.diff(wall))
print("dt_model=%.5f s   median real tick=%.5f s" % (dt_model, T_real))

# measured joint displacement per tick
dq_meas = np.diff(q_meas, axis=0)             # (N-1,6)
u_trim = u[:-1]                                # command that produced that step

# Expected displacement if robot executed u for one real tick
exp_step = u_trim * T_real

# ratio per joint over the run (use joints that actually move)
print("\n[EXECUTION] measured dq vs commanded u*dt")
print(" j   mean|dq_meas|   mean|u*dt|   exec_ratio(dq/u_dt)")
for j in range(6):
    a = np.abs(dq_meas[:, j]).mean()
    b = np.abs(exp_step[:, j]).mean()
    print("  %d   %.6f      %.6f      %.3f" % (j, a, b, a/b if b > 1e-9 else float('nan')))

# Commanded vs needed velocity:
# reference speed (what u SHOULD be to keep up) = |dq_ref|/dt
dq_ref = np.diff(q_ref, axis=0)
ref_speed = np.linalg.norm(dq_ref, axis=1) / dt_model
cmd_speed = np.linalg.norm(u_trim, axis=1)
print("\n[COMMAND vs NEED]")
print("  mean reference speed needed: %.4f rad/s" % ref_speed.mean())
print("  mean commanded |u|:          %.4f rad/s" % cmd_speed.mean())
print("  command/need ratio:          %.3f" % (cmd_speed.mean()/ref_speed.mean()))

# And does robot execute the (small) command it is given?
meas_speed = np.linalg.norm(dq_meas, axis=1) / T_real
print("\n[ROBOT vs COMMAND]")
print("  mean commanded |u|:    %.4f rad/s" % cmd_speed.mean())
print("  mean measured speed:   %.4f rad/s" % meas_speed.mean())
print("  measured/commanded:    %.3f" % (meas_speed.mean()/cmd_speed.mean()))
