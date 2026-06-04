import threading
import time
import numpy as np

from src.ur10e import UR10e
from src.mpc import MPC
from src.simulation import EmbeddedSimEnvironment


class MPCSimulationThread(threading.Thread):
    def __init__(self, shared_state, mpc_horizon=1, dt=0.01, vj=0.5, aj=1, workspace_offset=np.eye(4,4)):
        super().__init__(daemon=True)
        self.mpc_horizon = mpc_horizon
        self.results = {'t': None, 'y': None, 'u': None}
        self.status = "idle"  # idle, running, completed, error
        self.error_msg = None
        self.shared_state = shared_state
        self.dt = dt
        self.vj = vj
        self.aj = aj
        self.workspace_offset = workspace_offset

    def run(self):
        try:
            self.status = "running"
            print("[MPC Thread] Starting simulation...")

            # Q1
            ur10e = UR10e(dt=self.dt, workspace_offset=self.workspace_offset)

            # Instantiate controller
            x_lim, u_lim, acc_u_lim = ur10e.get_limits(self.vj, self.aj)

            # Create MPC Solver
            tracking_ctl = MPC(model=ur10e,
                               dynamics=ur10e.model,
                               param='P1',
                               N=self.mpc_horizon,
                               xlb=-x_lim, xub=x_lim,
                               ulb=-u_lim, uub=u_lim,
                               acc_ulb=-acc_u_lim, acc_uub=acc_u_lim,
                               shared_state=self.shared_state)

            sim_env_tracking = EmbeddedSimEnvironment(model=ur10e,
                                                      dynamics=ur10e.model,
                                                      controller=tracking_ctl.mpc_controller,
                                                      shared_state=self.shared_state)

            x0 = np.array(self.shared_state.home_joints)
            print(f"[MPC Thread] Starting from first trajectory point: {np.degrees(x0).round(1)}")
            t, y, u = sim_env_tracking.run(x0)

            self.results = {'t': t, 'y': y, 'u': u, 'env': sim_env_tracking}
            self.status = "completed"
            graph_number = np.random.randint(1, 1000)
            sim_env_tracking.visualize(graph_number)
            sim_env_tracking.visualize_error(graph_number)
            sim_env_tracking.visualize_end_effector(graph_number)
            print("[MPC Thread] Simulation completed successfully!")

        except Exception as e:
            self.status = "error"
            self.error_msg = str(e)
            print(f"[MPC Thread] Error: {e}")


def run_mpc_simulation(mpc_horizon=1, blocking=False):
    sim_thread = MPCSimulationThread(mpc_horizon=mpc_horizon)
    sim_thread.start()

    if blocking:
        sim_thread.join()
        if sim_thread.status == "error":
            raise RuntimeError(f"Simulation failed: {sim_thread.error_msg}")

    return sim_thread


if __name__ == "__main__":
    # Start the simulation in a background thread
    print("Starting MPC simulation in background thread...")
    sim_thread = run_mpc_simulation(mpc_horizon=1, blocking=False, )

    # You can do other work while simulation runs
    print("Main thread is free to do other work!")
    print("Waiting for simulation to complete...")

    # Wait for completion
    sim_thread.join()

    # Check results
    if sim_thread.status == "completed":
        print("\nVisualization results:")
        env = sim_thread.results['env']
        env.visualize()  # Visualize state propagation
        env.visualize_error()
    else:
        print(f"\nSimulation status: {sim_thread.status}")
        if sim_thread.error_msg:
            print(f"Error: {sim_thread.error_msg}")