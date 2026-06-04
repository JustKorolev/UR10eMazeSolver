"""
Model Predictive Control - CasADi interface
Adapted from Helge-Andre Langaker work on GP-MPC
Customized by Pedro Roque for EL2700 Model Predictive Countrol Course at KTH
"""
import time
import numpy as np
import casadi as ca
import casadi.tools as ctools
import yaml
import os
from src.utils import *


class MPC(object):

    def __init__(self, model, dynamics,
                 param='P1', N=10,
                 xlb=None, xub=None,
                 ulb=None, uub=None,
                 acc_ulb=None, acc_uub=None,
                 terminal_constraint=None, solver_opts=None,
                 tuning_file=None, shared_state=None):
        """
        Constructor for the MPC class.
        """

        # State flags
        build_solver_time = -time.time()
        self.model = model
        self.dt = model.dt
        self.Nx, self.Nu = model.n, model.m
        self.Nt = N
        print("Horizon steps: ", N * self.dt)
        self.dynamics = dynamics
        self.shared_state = shared_state

        # Initialize variables
        self.set_cost_functions()
        self.x_sp = None
        self.u_sp = None
        self.u_prev = np.zeros(self.Nu)

        # Cost function weights
        Q, R, P = self.load_params(param, tuning_file)

        self.Q = ca.MX(Q)
        self.P = ca.MX(P)
        self.R = ca.MX(R)

        if xub is None:
            xub = np.full((self.Nx), np.inf)
        if xlb is None:
            xlb = np.full((self.Nx), -np.inf)
        if uub is None:
            uub = np.full((self.Nu), np.inf)
        if ulb is None:
            ulb = np.full((self.Nu), -np.inf)
        if acc_uub is None:
            acc_uub = np.full((self.Nu), np.inf)
        if acc_ulb is None:
            acc_ulb = np.full((self.Nu), -np.inf)

        # Starting state parameters - add slack here
        x0 = ca.MX.sym('x0', self.Nx)
        x_ref = ca.MX.sym('x_ref', self.Nx*(self.Nt+1),)
        u_ref = ca.MX.sym('u_ref', self.Nu*self.Nt,)
        u0 = ca.MX.sym('u0', self.Nu)
        param_s = ca.vertcat(x0, x_ref, u0, u_ref)

        # Create optimization variables
        opt_var = ctools.struct_symMX([(ctools.entry('u', shape=(self.Nu,), repeat=self.Nt),
                                        ctools.entry('x', shape=(self.Nx,), repeat=self.Nt + 1),
                                        )])
        self.opt_var = opt_var
        self.num_var = opt_var.size

        # Decision variable boundries
        self.optvar_lb = opt_var(-np.inf)
        self.optvar_ub = opt_var(np.inf)

        # Set initial values
        obj = ca.MX(0)
        con_eq = []
        con_ineq = []
        con_ineq_lb = []
        con_ineq_ub = []
        con_eq.append(opt_var['x', 0] - x0)

        # Generate MPC Problem
        x_r_terminal = x_ref[self.Nx*self.Nt:(self.Nx)*(self.Nt+1)]
        for t in range(self.Nt):

            # Get variables
            x_t = opt_var['x', t]
            x_r = x_ref[self.Nx*t:(self.Nx)*(t+1)]
            u_r = u_ref[self.Nu*t:(self.Nu)*(t+1)]
            u_t = opt_var['u', t]

            # Dynamics constraint
            x_t_next = self.dynamics(x_t, u_t)
            con_eq.append(x_t_next - opt_var['x', t + 1])

            # Input constraints
            if uub is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(uub)
                con_ineq_lb.append(np.full((self.Nu,), -ca.inf))
            if ulb is not None:
                con_ineq.append(u_t)
                con_ineq_ub.append(np.full((self.Nu,), ca.inf))
                con_ineq_lb.append(ulb)

            # State constraints
            if xub is not None:
                con_ineq.append(x_t)
                con_ineq_ub.append(xub)
                con_ineq_lb.append(np.full((self.Nx,), -ca.inf))
            if xlb is not None:
                con_ineq.append(x_t)
                con_ineq_ub.append(np.full((self.Nx,), ca.inf))
                con_ineq_lb.append(xlb)

            if acc_uub is not None:
                if t == 0:
                    acc = (u_t - u0) / self.dt
                else:
                    acc = (u_t - opt_var['u', t - 1]) / self.dt

                con_ineq.append(acc)
                con_ineq_lb.append(acc_ulb)
                con_ineq_ub.append(acc_uub)

            # Objective Function / Cost Function
            obj += self.running_cost(x_t, x_r, self.Q, u_t, u_r, self.R)

        # Terminal Cost
        obj += self.terminal_cost(opt_var['x', self.Nt], x_r_terminal, self.P)

        # Terminal contraint
        if terminal_constraint is not None:
            # Should be a polytope
            H_N = terminal_constraint.A
            if H_N.shape[1] != self.Nx:
                print("Terminal constraint with invalid dimensions.")
                exit()

            H_b = terminal_constraint.b
            con_ineq.append(ca.mtimes(H_N, opt_var['x', self.Nt]))
            con_ineq_lb.append(-ca.inf * ca.DM.ones(H_N.shape[0], 1))
            con_ineq_ub.append(H_b)

        # Equality constraints bounds are 0 (they are equality constraints),
        # -> Refer to CasADi documentation
        num_eq_con = ca.vertcat(*con_eq).size1()
        num_ineq_con = ca.vertcat(*con_ineq).size1()
        con_eq_lb = np.zeros((num_eq_con,))
        con_eq_ub = np.zeros((num_eq_con,))

        # Set constraints
        con = ca.vertcat(*(con_eq + con_ineq))
        self.con_lb = ca.vertcat(con_eq_lb, *con_ineq_lb)
        self.con_ub = ca.vertcat(con_eq_ub, *con_ineq_ub)

        # Build NLP Solver (can also solve QP)
        solver_dict = dict(x=opt_var, f=obj, g=con, p=param_s)
        qp_opts = {
            'max_iter': 20,
            'error_on_fail': False,
            'print_header': False,
            'print_iter': False
        }

        options = {
            'max_iter': 5,
            'qpsol': 'qrqp',
            'qpsol_options': qp_opts,
            'convexify_margin': 1e-4,
            'print_header': False,
            'print_iteration': False,
            'print_status': False,
            'print_time': False,
            'error_on_fail': False,
        }
        if solver_opts is not None:
            options.update(solver_opts)
        self.solver = ca.nlpsol('mpc_solver', 'sqpmethod', solver_dict, options)

        # u_ref_solver initialization
        u_ref_i = ca.MX.sym('u_ref_i', self.Nu)
        x_ref_i = ca.MX.sym('x_ref_i', self.Nx)
        x_ref_i_next = ca.MX.sym('x_ref_i_next', self.Nx)
        err = self.dynamics(x_ref_i, u_ref_i) - x_ref_i_next
        J = ca.dot(err, err)

        solver_dict = {'x': u_ref_i, 'f': J, 'p': ca.vertcat(x_ref_i, x_ref_i_next)}
        self.u_ref_solver = ca.nlpsol('u_ref_solver', 'sqpmethod', solver_dict, options)

        build_solver_time += time.time()
        print('\n________________________________________')
        print('# Time to build mpc solver: %f sec' % build_solver_time)
        print('# Number of variables: %d' % self.num_var)
        print('# Number of equality constraints: %d' % num_eq_con)
        print('# Number of inequality constraints: %d' % num_ineq_con)
        print('----------------------------------------')
        pass

    def load_params(self, param, tuning_file=None):
        """
        Parameters loader function.
        Loads yaml parameters to generate Q, R and P.

        :param param: parameter setting ('P1', 'P2', 'P3')
        :type param: string
        """

        if param not in ['P1', 'P2', 'P3']:
            print("Wrong param option. Select param='P1' or 'P2' or 'P3'.")
            exit()

        if tuning_file is None:
            f_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../resources/tuning.yaml")
        else:
            f_path = tuning_file

        with open(f_path, 'r') as stream:
            parameters = yaml.safe_load(stream)

            # Create numpy diags
            Q_diag = np.asarray(parameters[param]['Q'])
            R_diag = np.asarray(parameters[param]['R'])
            P_diag = parameters[param]['P_mult'] * Q_diag

            # Get matrices
            Q = np.diag(Q_diag)
            R = np.diag(R_diag)
            P = np.diag(P_diag)

            return Q, R, P

        raise NotADirectoryError("Wrong directory for yaml file.")

    def set_cost_functions(self):
        """
        Helper method to create CasADi functions for the MPC cost objective.
        """
        # Create functions and function variables for calculating the cost
        Q = ca.MX.sym('Q', self.Nx, self.Nx)
        R = ca.MX.sym('R', self.Nu, self.Nu)
        P = ca.MX.sym('P', self.Nx, self.Nx)

        x = ca.MX.sym('x', self.Nx)
        xr = ca.MX.sym('xr', self.Nx)
        u = ca.MX.sym('u', self.Nu)
        ur = ca.MX.sym('ur',self.Nu)

        # Prepare variables
        q = x[0:6]
        qdot = u[0:6]

        qr = xr[0:6]
        qdotr = ur[0:6]

        # Calculate errors
        eq = wrap_joints_cas(q - qr)
        eqdot = qdot - qdotr

        xe_vec = ca.vertcat(eq)
        ue_vec = ca.vertcat(eqdot)

        # Calculate running cost
        ln = ca.mtimes(ca.mtimes(xe_vec.T, Q), xe_vec) \
            + ca.mtimes(ca.mtimes(ue_vec.T, R), ue_vec)

        self.running_cost = ca.Function('ln', [x, xr, Q, u, ur, R], [ln])

        # Calculate terminal cost
        V = ca.mtimes(ca.mtimes(xe_vec.T, P), xe_vec)
        self.terminal_cost = ca.Function('V', [x, xr, P], [V])

    def solve_mpc(self, x0, u0=None):
        """
        Solve the optimal control problem

        :param x0: starting state
        :type x0: np.ndarray
        :param u0: optimal control guess, defaults to None
        :type u0: np.ndarray, optional
        :return: predicted optimal states and optimal control inputs
        :rtype: ca.DM
        """

        # Initial state
        if u0 is None:
            u0 = np.zeros(self.Nu)
        if self.x_sp is None:
            self.x_sp = np.zeros(self.Nx * (self.Nt + 1))

        # Initialize variables
        self.optvar_x0 = np.full((1, self.Nx), x0.T)

        # Initial guess of the warm start variables
        self.optvar_init = self.opt_var(0)
        self.optvar_init['x', 0] = self.optvar_x0[0]

        solve_time = -time.time()

        param = ca.vertcat(x0, self.x_sp, u0, self.u_sp)
        args = dict(x0=self.optvar_init,
                    lbx=self.optvar_lb,
                    ubx=self.optvar_ub,
                    lbg=self.con_lb,
                    ubg=self.con_ub,
                    p=param)

        # Solve NLP
        sol = self.solver(**args)
        status = self.solver.stats()['return_status']
        optvar = self.opt_var(sol['x'])

        solve_time += time.time()
        # print('MPC - CPU time: %f seconds  |  Cost: %f  |  Horizon length: %d ' % (solve_time, sol['f'], self.Nt))
        if status == "Infeasible_Problem_Detected":
            print("Infeasible_Problem_Detected")
            exit()

        return optvar['x'], optvar['u']

    def mpc_controller(self, x0, t, prerecorded=False):
        """
        MPC controller wrapper.
        Gets first control input to apply to the system.

        :param x0: initial state
        :type x0: np.ndarray
        :return: control input
        :rtype: ca.DM
        """

        if prerecorded:
            try:
                x_traj = self.model.get_joint_trajectory(t, self.Nt + 1)
                x_sp = x_traj.reshape(self.Nx * (self.Nt + 1), order="F")
            except IndexError:
                self.shared_state.stop_following()
        else:
            x_traj = np.array(self.shared_state.trajectory_window)
            x_sp = x_traj.reshape(self.Nx * (self.Nt + 1), order="C")
        self.set_reference(x_sp)


        _, u_pred = self.solve_mpc(x0, u0=self.u_prev)

        self.shared_state.consume_one()

        u_cmd = np.array(u_pred[0])
        self.u_prev = u_cmd.copy()

        error = self.calculate_error(x0, self.x_sp[0:6])
        return u_cmd, error

    def calculate_error(self, x, xr):
        """
        Calculate error, used for logging

        :param x: [description]
        :type x: [type]
        :param xr: [description]
        :type xr: [type]
        :param u: [description]
        :type u: [type]
        :param ur: [description]
        :type ur: [type]
        :return: [description]
        :rtype: [type]
        """

        # Prepare xr
        xr = xr.reshape((xr.shape[0], 1))

        # Prepare variables
        q = x[0:6]
        qr = xr[0:6]

        # Calculate errors
        xeq = wrap_joints_np(q - qr)

        return xeq.reshape((6, 1))

    def set_reference(self, x_sp):
        """
        Set the controller reference state

        :param x_sp: desired reference state
        :type x_sp: np.ndarray
        """
        self.x_sp = x_sp

        u_sp = np.zeros(self.Nu*self.Nt)
        for i in range(self.Nt):
            param = ca.vertcat(x_sp[self.Nx*i:self.Nx*(i+1)],
                            x_sp[self.Nx*(i+1):self.Nx*(i+2)])
            sol = self.u_ref_solver(x0=np.zeros(self.Nu), p=param)
            u_ref_i = np.array(sol['x']).reshape(-1, 1)
            u_sp[self.Nu*i:self.Nu*(i+1)] = u_ref_i.flatten()

        u_sp = np.zeros(self.Nu*self.Nt)
        self.u_sp = u_sp
