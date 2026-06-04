import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import time


def _z_nullspace_project(model, x_mod, u, workspace_z,
                         k_restore=1.0, z_deadband=0.003,
                         max_correction=0.15, eps=1e-5):
    """Minimally modify joint velocity *u* so the end-effector Z velocity
    becomes ``-k_restore * (z_cur - workspace_z)`` instead of whatever the
    MPC produced.  This both prevents new Z drift and corrects existing drift.

    A deadband avoids jitter when Z is already close to the target, and
    the correction magnitude is capped for stability on the real robot.
    """
    x_mod = np.asarray(x_mod, dtype=float).reshape(6,)
    u_flat = np.asarray(u, dtype=float).flatten()

    theta_class_deg = np.rad2deg(model.DHModifiedToClassical(x_mod))
    T_ref = model.FK(theta_class_deg)
    z_cur = T_ref[2, 3]

    z_err = z_cur - workspace_z

    J_z = np.zeros(6)
    for i in range(6):
        x_pert = x_mod.copy()
        x_pert[i] += eps
        theta_pert_deg = np.rad2deg(model.DHModifiedToClassical(x_pert))
        T_pert = model.FK(theta_pert_deg)
        J_z[i] = (T_pert[2, 3] - z_cur) / eps

    J_z_norm_sq = np.dot(J_z, J_z)
    if J_z_norm_sq < 1e-10:
        return u

    vz_current = np.dot(J_z, u_flat)

    if abs(z_err) < z_deadband:
        vz_desired = 0.0
    else:
        vz_desired = -k_restore * z_err

    correction_scale = (vz_desired - vz_current) / J_z_norm_sq
    correction = J_z * correction_scale

    corr_norm = np.linalg.norm(correction)
    if corr_norm > max_correction:
        correction *= max_correction / corr_norm

    u_proj = u_flat + correction
    return u_proj.reshape(np.asarray(u).shape)


class EmbeddedSimEnvironment(object):

    def __init__(self, model, dynamics, controller, time=100.0, shared_state=None):
        """
        Embedded simulation environment. Simulates the syste given dynamics
        and a control law, plots in matplotlib.

        :param model: model object
        :type model: object
        :param dynamics: system dynamics function (x, u)
        :type dynamics: casadi.DM
        :param controller: controller function (x, r)
        :type controller: casadi.DM
        :param time: total simulation time, defaults to 100 seconds
        :type time: float, optional
        """
        self.model = model
        self.dynamics = dynamics
        self.controller = controller
        self.dt = self.model.dt
        self.estimation_in_the_loop = False
        self.shared_state = shared_state
        self.ran_iterations = 0

    def run(self, x0):
        """
        Run simulator with specified system dynamics and control function.
        """

        print("Running simulation....")
        t = np.array([0])
        x_vec = np.array([x0]).reshape(self.model.n, 1)
        u_vec = np.empty((6, 0))
        e_vec = np.empty((6, 0))

        start_wall = time.time()
        next_tick = start_wall

        self.ran_iterations = 0
        while self.shared_state.following_trajectory:
            loop_start = time.time()

            x_measured = np.array(self.shared_state.joint_pos)
            if x_measured is None:
                continue
            x = self.model.DHClassicaltoModified(x_measured).reshape((self.model.n, 1)) # TODO: UNCOMMENT THIS FOR ACTUAL ROBOT
            # print("curr_state")
            # print(x)
            # x = x_vec[:, -1].reshape((self.model.n, 1)) # TODO: REMOVE THIS FOR ACTUAL ROBOT
            # print("predicted_state")
            # print(x)
            u, error = self.controller(x, self.ran_iterations * self.dt, prerecorded=self.shared_state.prerecorded_flag)

            if hasattr(self.shared_state, '_workspace_z'):
                u = _z_nullspace_project(
                    self.model, x.flatten(), u,
                    self.shared_state._workspace_z
                )

            x_next = self.dynamics(x, u)

            with self.shared_state.lock:
                self.shared_state.u_curr = np.array(u).reshape(6, 1)

            t = np.append(t, t[-1] + self.dt)
            x_vec = np.append(x_vec, np.array(x_next).reshape(self.model.n, 1), axis=1)
            u_vec = np.append(u_vec, np.array(u).reshape(self.model.m, 1), axis=1)
            e_vec = np.append(e_vec, error.reshape(6, 1), axis=1)

            self.ran_iterations += 1

            next_tick += self.dt
            sleep_time = next_tick - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)

        _, error = self.controller(x_next, self.ran_iterations * self.dt)
        e_vec = np.append(e_vec, error.reshape((6, 1)), axis=1)

        self.t = t
        self.x_vec = x_vec
        self.u_vec = u_vec
        self.e_vec = e_vec
        return t, x_vec, u_vec

    def visualize(self, number):
        """
        Offline plotting of simulation data for UR10e joint-space model.
        State: q in R^6
        Input: qdot in R^6
        """
        loop_length = self.ran_iterations / self.dt
        variables = [self.t, self.x_vec, self.u_vec, loop_length]
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")
            return

        t = self.t
        x_vec = self.x_vec
        u_vec = self.u_vec

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Joint positions
        ax1.clear()
        ax1.set_title("UR10e Joint States")
        for i in range(6):
            ax1.plot(t, x_vec[i, :], '--', label=f"q{i+1}")
        ax1.set_ylabel("Joint Angle [rad]")
        ax1.grid()
        ax1.legend()

        # Joint velocities
        ax2.clear()
        ax2.set_title("UR10e Joint Velocity Commands")
        for i in range(6):
            ax2.plot(t[:-1], u_vec[i, :], '--', label=f"qdot{i+1}")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Joint Velocity [rad/s]")
        ax2.grid()
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"./tracking_diagnosis/diagnosis{number}.png")

    def visualize_error(self, number):
        """
        Offline plotting of tracking error for UR10e joint-space model.
        Error: q - q_ref in R^6
        Input: qdot in R^6
        """
        loop_length = self.ran_iterations / self.dt
        variables = [self.t, self.e_vec, self.u_vec, loop_length]
        if any(elem is None for elem in variables):
            print("Please run the simulation first with the method 'run'.")
            return

        t = self.t
        e_vec = self.e_vec
        u_vec = self.u_vec

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Joint position tracking error
        ax1.clear()
        ax1.set_title("UR10e Joint Tracking Error")
        for i in range(6):
            ax1.plot(t, e_vec[i, :], '--', label=f"e_q{i+1}")
        ax1.set_ylabel("Joint Error [rad]")
        ax1.grid()
        ax1.legend()

        # Joint velocity commands
        ax2.clear()
        ax2.set_title("UR10e Joint Velocity Commands")
        for i in range(6):
            ax2.plot(t[:-1], u_vec[i, :], '--', label=f"qdot{i+1}")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Joint Velocity [rad/s]")
        ax2.grid()
        ax2.legend()

        plt.tight_layout()
        plt.savefig(f"./tracking_diagnosis/error_diagnosis{number}.png")

    def visualize_end_effector(self, number):
        """Compute FK for every simulated joint state and plot end-effector XYZ."""
        if self.ran_iterations == 0:
            print("Please run the simulation first with the method 'run'.")
            return

        t = self.t
        x_vec = self.x_vec
        n_pts = x_vec.shape[1]

        ee_pos = np.zeros((3, n_pts))
        for k in range(n_pts):
            theta_mod_rad = x_vec[:, k]
            theta_class_rad = self.model.DHModifiedToClassical(theta_mod_rad)
            theta_class_deg = np.rad2deg(theta_class_rad)
            T = self.model.FK(theta_class_deg)
            ee_pos[:, k] = T[:3, 3]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))

        ax_xyz = axes[0, 0]
        ax_xyz.set_title("End-Effector Position vs Time")
        ax_xyz.plot(t, ee_pos[0, :], '-', label="X")
        ax_xyz.plot(t, ee_pos[1, :], '-', label="Y")
        ax_xyz.plot(t, ee_pos[2, :], '-', label="Z")
        ax_xyz.set_xlabel("Time [s]")
        ax_xyz.set_ylabel("Position [m]")
        ax_xyz.grid()
        ax_xyz.legend()

        ax_xy = axes[0, 1]
        ax_xy.set_title("End-Effector XY (Top View)")
        ax_xy.plot(ee_pos[0, :], ee_pos[1, :], '-')
        ax_xy.scatter(ee_pos[0, 0], ee_pos[1, 0], c='green', s=80, zorder=5, label="Start")
        ax_xy.scatter(ee_pos[0, -1], ee_pos[1, -1], c='red', s=80, zorder=5, label="End")
        ax_xy.set_xlabel("X [m]")
        ax_xy.set_ylabel("Y [m]")
        ax_xy.axis('equal')
        ax_xy.grid()
        ax_xy.legend()

        ax_xz = axes[1, 0]
        ax_xz.set_title("End-Effector XZ (Front View)")
        ax_xz.plot(ee_pos[0, :], ee_pos[2, :], '-')
        ax_xz.scatter(ee_pos[0, 0], ee_pos[2, 0], c='green', s=80, zorder=5, label="Start")
        ax_xz.scatter(ee_pos[0, -1], ee_pos[2, -1], c='red', s=80, zorder=5, label="End")
        ax_xz.set_xlabel("X [m]")
        ax_xz.set_ylabel("Z [m]")
        ax_xz.axis('equal')
        ax_xz.grid()
        ax_xz.legend()

        ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
        axes[1, 1].remove()
        ax_3d.set_title("End-Effector 3D Trajectory")
        ax_3d.plot(ee_pos[0, :], ee_pos[1, :], ee_pos[2, :], '-')
        ax_3d.scatter(*ee_pos[:, 0], c='green', s=80, label="Start")
        ax_3d.scatter(*ee_pos[:, -1], c='red', s=80, label="End")
        ax_3d.set_xlabel("X [m]")
        ax_3d.set_ylabel("Y [m]")
        ax_3d.set_zlabel("Z [m]")
        ax_3d.legend()

        plt.tight_layout()
        plt.savefig(f"./tracking_diagnosis/end_effector{number}.png")
