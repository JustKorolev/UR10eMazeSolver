import numpy as np


class JointLimitCBF:
    """CBF constraints that keep joint positions away from position limits."""

    def __init__(self, q_min, q_max, margin=0.05):
        self.q_min = np.asarray(q_min, dtype=float).reshape(-1)
        self.q_max = np.asarray(q_max, dtype=float).reshape(-1)
        self.margin = float(margin)

        if self.q_min.shape != self.q_max.shape:
            raise ValueError("q_min and q_max must have the same shape")

    def inequalities(self, q, alpha):
        """
        Return A, b for CBF inequalities A @ u >= b.

        Lower limit:
            h(q) = q - q_min - margin >= 0
            dh/dq @ u >= -alpha h

        Upper limit:
            h(q) = q_max - margin - q >= 0
            dh/dq @ u >= -alpha h
        """
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape != self.q_min.shape:
            raise ValueError("q must have the same shape as joint limits")

        A = []
        b = []

        for i in range(len(q)):
            lower_row = np.zeros(len(q))
            lower_row[i] = 1.0
            lower_h = q[i] - self.q_min[i] - self.margin
            A.append(lower_row)
            b.append(-alpha * lower_h)

            upper_row = np.zeros(len(q))
            upper_row[i] = -1.0
            upper_h = self.q_max[i] - self.margin - q[i]
            A.append(upper_row)
            b.append(-alpha * upper_h)

        return np.asarray(A, dtype=float), np.asarray(b, dtype=float)


class CBFSafetyFilter:
    """
    Post-MPC CBF-QP safety filter.

    Solves:
        min_u 0.5 ||u - u_des||^2
        s.t.  A_i(q) u >= b_i(q)
              u_min <= u <= u_max
    """

    def __init__(
        self,
        constraints=None,
        alpha=5.0,
        u_min=None,
        u_max=None,
        projection_iterations=30,
    ):
        self.constraints = list(constraints or [])
        self.alpha = float(alpha)
        self.u_min = None if u_min is None else np.asarray(u_min, dtype=float).reshape(-1)
        self.u_max = None if u_max is None else np.asarray(u_max, dtype=float).reshape(-1)
        self.projection_iterations = int(projection_iterations)

    def filter(self, q, u_des):
        q = np.asarray(q, dtype=float).reshape(-1)
        original_shape = np.asarray(u_des).shape
        u_des = np.asarray(u_des, dtype=float).reshape(-1)

        if q.shape != u_des.shape:
            raise ValueError("q and u_des must have the same flattened shape")

        A, b = self._build_inequalities(q)
        u_safe = self._solve_qp(u_des, A, b)
        return u_safe.reshape(original_shape)

    def _build_inequalities(self, q):
        A_blocks = []
        b_blocks = []

        for constraint in self.constraints:
            A_i, b_i = constraint.inequalities(q, self.alpha)
            A_blocks.append(A_i)
            b_blocks.append(b_i)

        if not A_blocks:
            return np.empty((0, len(q))), np.empty((0,))

        return np.vstack(A_blocks), np.concatenate(b_blocks)

    def _solve_qp(self, u_des, A, b):
        u_cvx = self._solve_with_cvxpy(u_des, A, b)
        if u_cvx is not None:
            return u_cvx
        return self._solve_with_projection(u_des, A, b)

    def _solve_with_cvxpy(self, u_des, A, b):
        try:
            import cvxpy as cp
        except ImportError:
            return None

        u = cp.Variable(len(u_des))
        constraints = []
        if A.size > 0:
            constraints.append(A @ u >= b)
        if self.u_min is not None:
            constraints.append(u >= self.u_min)
        if self.u_max is not None:
            constraints.append(u <= self.u_max)

        problem = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(u - u_des)), constraints)

        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except Exception:
            return None

        if u.value is None:
            return None

        return np.asarray(u.value, dtype=float).reshape(-1)

    def _solve_with_projection(self, u_des, A, b):
        """Dykstra-style projection fallback for the box/halfspace QP."""
        u = u_des.copy()

        if self.u_min is not None:
            u = np.maximum(u, self.u_min)
        if self.u_max is not None:
            u = np.minimum(u, self.u_max)

        if A.size == 0:
            return u

        corrections = np.zeros_like(A)

        for _ in range(max(1, self.projection_iterations)):
            for i, a in enumerate(A):
                y = u + corrections[i]
                violation = b[i] - float(a @ y)

                if violation > 0:
                    denom = float(a @ a)
                    if denom > 1e-12:
                        projected = y + (violation / denom) * a
                    else:
                        projected = y
                else:
                    projected = y

                corrections[i] = y - projected
                u = projected

                if self.u_min is not None:
                    u = np.maximum(u, self.u_min)
                if self.u_max is not None:
                    u = np.minimum(u, self.u_max)

        return u
