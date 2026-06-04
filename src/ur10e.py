from typing import TypedDict
import numpy as np
import src.utils as utils
import os

class UR10e():
    def __init__(self, dt=0.01, workspace_offset=np.eye(4,4)):
        # Kinematics
        self.dof = 6
        self.link_lengths = [0.1273, 0.612, 0.5723, 0.163941, 0.1157, 0.0922] # in meters
        self.Tb0 = utils.trans_z(0.1807)  # Base to first joint transformation
        self.T6tp = utils.trans_z(0.11655)

        # Dynamics
        self.model = self.dynamics
        self.n = 6
        self.m = 6
        self.dt = dt

        self.pose_trajectory = None
        self.joint_trajectory = None
        self.workspace_offset = workspace_offset

    class DHParameters(TypedDict):
        theta: float  # in degrees
        d: float      # in meters
        a: float      # in meters
        alpha: float  # in degrees

    class ModifiedDHParameters(TypedDict):
        a_i_prev: float  # in degrees
        alpha_i_prev: float  # in meters
        d_i: float  # in meters
        theta_i: float  # in degrees

    def get_limits(self, vj, aj):
        x_lim = 2*np.pi * np.ones(6) # radians
        u_lim = vj * np.ones(6) # radians/sec
        acc_u_lim = aj * np.ones(6) # radians/sec²

        return x_lim, u_lim, acc_u_lim

    def get_classical_dh_parameters(self, joint_angles) -> DHParameters:
        alpha = [90, 0, 0, 90, -90, 0]
        a = [0.0, -0.6127, -0.57155, 0.0, 0.0, 0.0]
        d = [0.1807, 0.0, 0.0, 0.17415, 0.11985, 0.11655]
        theta = joint_angles
        return {'theta': theta, 'd': d, 'a': a, 'alpha': alpha}

    def get_modified_dh_parameters(self, joint_angles) -> ModifiedDHParameters:
        alpha_i_prev = [0, 90, 0, 0, -90, 90] # in degrees
        a_i_prev = [0, 0, 0.6127, 0.57155, 0, 0] # in meters
        d_i = [0, 0, 0, 0.17415, 0.11985, 0] # in meters
        theta_i = [joint_angles[0], joint_angles[1]+90, joint_angles[2],
                   joint_angles[3]-90, joint_angles[4], joint_angles[5]] # in degrees

        return {
            'a_i_prev': a_i_prev,
            'alpha_i_prev': alpha_i_prev,
            'd_i': d_i,
            'theta_i': theta_i,
        }

    def DHModifiedToClassical(self, theta_mod):
        theta_mod = np.asarray(theta_mod, dtype=float).reshape(6,)

        home_class = np.deg2rad([0, -90, 0, -90, 0, 180])

        theta_class = theta_mod + home_class

        return theta_class

    def DHClassicaltoModified(self, theta_class):
        theta_class = np.asarray(theta_class, dtype=float).reshape(6,)

        home_class = np.deg2rad([0, -90, 0, -90, 0, 180])

        theta_mod = theta_class - home_class

        return theta_mod

    def IK(self, solution_type: str, T_base_tp: np.ndarray, Ttp_pen: np.ndarray = utils.trans_z(0.3)) -> np.ndarray:

        dh_parameters = self.get_modified_dh_parameters([0, 0, 0, 0, 0, 0])  # Placeholder joint angles for DH parameters
        a = dh_parameters['a_i_prev']
        d = dh_parameters['d_i']

        T_06 = np.linalg.inv(self.Tb0) @ T_base_tp @ np.linalg.inv(self.T6tp)

        if Ttp_pen is not None:
             T_06 = T_06 @ np.linalg.inv(Ttp_pen)

        # Retrieve components from the transformation matrix
        r11 = T_06[0, 0]
        r12 = T_06[0, 1]
        r13 = T_06[0, 2]
        r21 = T_06[1, 0]
        r22 = T_06[1, 1]
        r23 = T_06[1, 2]
        r31 = T_06[2, 0]
        r32 = T_06[2, 1]
        r33 = T_06[2, 2]
        px = T_06[0, 3]
        py = T_06[1, 3]
        pz = T_06[2, 3]

        # Calculate joint angles
        E1, F1, G1 = py, -px, d[3]

        t1_pos = (-F1 + np.sqrt(E1**2 + F1**2 - G1**2)) / (G1 - E1)
        t1_neg = (-F1 - np.sqrt(E1**2 + F1**2 - G1**2)) / (G1 - E1)

        theta_1_1 = 2 * np.arctan(t1_pos)
        theta_1_2 = 2 * np.arctan(t1_neg)

        theta_6_1 = np.arctan2(r12 * np.sin(theta_1_1) - r22 * np.cos(theta_1_1),
                            r21 * np.cos(theta_1_1) - r11 * np.sin(theta_1_1))
        theta_6_2 = np.arctan2(r12 * np.sin(theta_1_2) - r22 * np.cos(theta_1_2),
                            r21 * np.cos(theta_1_2) - r11 * np.sin(theta_1_2))

        theta_5_1 = np.arctan2((r21 * np.cos(theta_1_1) - r11 * np.sin(theta_1_1)) * np.cos(theta_6_1)
                            + (r12 * np.sin(theta_1_1) - r22 * np.cos(theta_1_1)) * np.sin(theta_6_1),
                            r13 * np.sin(theta_1_1) - r23 * np.cos(theta_1_1))
        theta_5_2 = np.arctan2((r21 * np.cos(theta_1_2) - r11 * np.sin(theta_1_2)) * np.cos(theta_6_2)
                            + (r12 * np.sin(theta_1_2) - r22 * np.cos(theta_1_2)) * np.sin(theta_6_2),
                            r13 * np.sin(theta_1_2) - r23 * np.cos(theta_1_2))

        A1 = (r31 * np.cos(theta_6_1) - r32 * np.sin(theta_6_1)) / np.cos(theta_5_1)
        A2 = (r31 * np.cos(theta_6_2) - r32 * np.sin(theta_6_2)) / np.cos(theta_5_2)
        B1 = (r31 * np.sin(theta_6_1) + r32 * np.cos(theta_6_1))
        B2 = (r31 * np.sin(theta_6_2) + r32 * np.cos(theta_6_2))

        k1 = -px * np.cos(theta_1_1) - py * np.sin(theta_1_1) - d[4] * A1
        k2 = -px * np.cos(theta_1_2) - py * np.sin(theta_1_2) - d[4] * A2
        b1 = pz - d[4] * B1
        b2 = pz - d[4] * B2

        E2_1 = -2 * a[2] * b1
        E2_2 = -2 * a[2] * b2
        F2_1 = -2 * a[2] * k1
        F2_2 = -2 * a[2] * k2
        G2_1 = a[2]**2 + k1**2 + b1**2 - a[3]**2
        G2_2 = a[2]**2 + k2**2 + b2**2 - a[3]**2

        t2_pos_1 = (-F2_1 + np.sqrt(F2_1**2 + E2_1**2 - G2_1**2)) / (G2_1 - E2_1)
        t2_neg_1 = (-F2_1 - np.sqrt(F2_1**2 + E2_1**2 - G2_1**2)) / (G2_1 - E2_1)
        t2_pos_2 = (-F2_2 + np.sqrt(F2_2**2 + E2_2**2 - G2_2**2)) / (G2_2 - E2_2)
        t2_neg_2 = (-F2_2 - np.sqrt(F2_2**2 + E2_2**2 - G2_2**2)) / (G2_2 - E2_2)

        theta_2_1 = 2 * np.arctan(t2_pos_1)
        theta_2_2 = 2 * np.arctan(t2_neg_1)
        theta_2_3 = 2 * np.arctan(t2_pos_2)
        theta_2_4 = 2 * np.arctan(t2_neg_2)

        theta_3_1 = np.arctan2(k1 - a[2] * np.sin(theta_2_1), b1 - a[2] * np.cos(theta_2_1)) - theta_2_1
        theta_3_2 = np.arctan2(k1 - a[2] * np.sin(theta_2_2), b1 - a[2] * np.cos(theta_2_2)) - theta_2_2
        theta_3_3 = np.arctan2(k2 - a[2] * np.sin(theta_2_3), b2 - a[2] * np.cos(theta_2_3)) - theta_2_3
        theta_3_4 = np.arctan2(k2 - a[2] * np.sin(theta_2_4), b2 - a[2] * np.cos(theta_2_4)) - theta_2_4

        theta_4_1 = np.arctan2(A1, B1) - theta_2_1 - theta_3_1
        theta_4_2 = np.arctan2(A1, B1) - theta_2_2 - theta_3_2
        theta_4_3 = np.arctan2(A2, B2) - theta_2_3 - theta_3_3
        theta_4_4 = np.arctan2(A2, B2) - theta_2_4 - theta_3_4

        sol0 = np.array([theta_1_1, theta_2_1, theta_3_1, theta_4_1, theta_5_1, theta_6_1])
        sol1 = np.array([theta_1_1, theta_2_2, theta_3_2, theta_4_2, theta_5_1, theta_6_1])
        sol2 = np.array([theta_1_2, theta_2_3, theta_3_3, theta_4_3, theta_5_2, theta_6_2])
        sol3 = np.array([theta_1_2, theta_2_4, theta_3_4, theta_4_4, theta_5_2, theta_6_2])

        if solution_type == 'elbow_up':
            return sol0
        elif solution_type == 'elbow_down':
            return sol1
        elif solution_type == 'elbow_up_2':
            return sol2
        elif solution_type == 'elbow_down_2':
            return sol3

    def jacobian(self, q):
        pass

    def FK(self, theta_1_6, Ttp_pen = utils.trans_z(0.3)) -> np.ndarray:
        theta_deg = np.asarray(theta_1_6, dtype=float).reshape(6,)

        dh = self.get_classical_dh_parameters(theta_deg)

        a = np.asarray(dh["a"], dtype=float)
        d = np.asarray(dh["d"], dtype=float)
        alpha_deg = dh["alpha"]
        theta_deg = dh["theta"]

        T = np.eye(4)
        for i in range(6):
            T = T @ utils.dh_classical_tf(a[i], utils.deg2rad(alpha_deg[i]), d[i], utils.deg2rad(theta_deg[i]))

        if Ttp_pen is not None:
            T = T @ Ttp_pen

        return T

    def dynamics(self, q, qdot):
        return q + self.dt * qdot

    def get_joint_trajectory(self, t, npoints):
        """
        Provide trajectory to be followed.
        :param t0: starting time
        :type t0: float
        :param npoints: number of trajectory points
        :type npoints: int
        :return: trajectory with shape (Nx, npoints)
        :rtype: np.array
        """

        # Current saved trajectories are poses
        if t == 0.0:
            f_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../trajectories/traj_23.txt") # TODO: CHANGE TRAJECTORY HERE
            print(f_path)
            self.pose_trajectory = np.loadtxt(f_path, ndmin=2)[:,1:]
            print((self.n, int(self.pose_trajectory.shape[0])))

            # Pose to joint trajectory conversion
            joint_trajectory = []
            for pose6 in self.pose_trajectory:
                pose_T = self.workspace_offset @ utils.pose6_to_T(pose6) # TODO: FIX THIS PRE TRASNFORMATION
                joints = self.IK("elbow_up_2", pose_T)
                joint_trajectory.append(joints)

            joint_trajectory = np.array(joint_trajectory)
            self.trajectory = joint_trajectory.T

        id_s = int(round(t / self.dt))
        id_e = int(round(t / self.dt)) + npoints
        x_r = self.trajectory[:, id_s:id_e]

        return x_r

    def get_initial_pose(self):
        """
        Helper function to get a starting state, depending on the dynamics type.

        :return: starting state
        :rtype: np.ndarray
        """
        T = self.workspace_offset
        x0 = self.IK("elbow_up_2", T)
        print(x0)
        return x0

if __name__ == "__main__":
    robot = UR10e()

    T_base_tip_example = utils.pose6_to_T([0.5, 0.7, 0.2, np.pi/8, np.pi/2, np.pi/4])
    Ttp_pen_example = utils.trans_z(0.3)
    desired_pose = utils.T_to_pose6(T_base_tip_example)
    print("Desired T_base_tip:\n",T_base_tip_example)
    print("Desired pose:\n", desired_pose)
    print()

    joint_angles_solution = robot.IK('elbow_down_2', T_base_tip_example, Ttp_pen_example)
    print("Joint angles for modified DH parameters solution:\n", joint_angles_solution)
    print()

    classical_joint_angles = np.rad2deg(robot.DHModifiedToClassical(joint_angles_solution))
    print("Joint angles in classical DH parameters:\n", classical_joint_angles)
    print()

    T = robot.FK(classical_joint_angles, Ttp_pen=Ttp_pen_example)
    end_pose = utils.T_to_pose6(T)

    safe = utils.SafetyCheck(robot, classical_joint_angles, Ttp_pen_example)
    print("End-effector transform from FK: \n", T)
    print("End-effector pose from FK:\n", end_pose)
    print()

    print("Is the solution safe?", safe)
    print("IK solution is valid:", np.allclose(T, T_base_tip_example, atol=1e-3))