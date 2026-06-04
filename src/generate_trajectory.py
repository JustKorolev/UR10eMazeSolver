import numpy as np

dt = 0.1
T = 15.0
t = np.arange(0.0, T + dt, dt)   # 151 points
N = len(t)

# Start and end joint configurations [rad]
q_start = np.array([
    0.0,
    -1.5707963267948966,
    1.5707963267948966,
    -1.5707963267948966,
    -1.5707963267948966,
    0.0
])

q_end = np.array([
    1,
    0,
    -1.2307963267948966,
    -1.2607963267948965,
    -1.4007963267948965,
    1
])

# Minimum-jerk scaling: s in [0,1]
tau = t / T
s = 10*tau**3 - 15*tau**4 + 6*tau**5

# Build trajectory matrix of shape (6, N)
traj = q_start[:, None] + (q_end - q_start)[:, None] * s[None, :]

# Save in the flat column format your loader expects
flat = traj.reshape(-1, order="F")
np.savetxt("resources/trajectory.txt", flat, fmt="%.18e")