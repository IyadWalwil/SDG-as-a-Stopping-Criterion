import numpy as np

class OptimizationMeasures:
    """
    Class for computing the optimization measures we study:
        * Optimality gap.
        * Feasibility gap. 
        * KKT error.
        * Smoothed duality gap. 
        * Projected duality gap.
    """

    def __init__(self, f=None, prox_f=None, stationarity=None, fc=None, proj_fc=None,
                 f_star=None, A=None, b=None):
        """
        Initializes an instance of the OptimizationMeasures class.

        Attributes:
            f (function): Objective function.
            prox_f (function): Proximal operator of the objective function.
            stationarity (function): Stationarity of the Lagrangian function.
            fc (function): Fenchel-Conjugate of the objective function.
            proj_fc (function): Projection onto the domain of fc.
            f_star (float): Optimal value of the objective function.
            A (numpy.ndarray): Coefficient matrix of the linear constraints.
            b (numpy.ndarray): Right-hand side vector of the linear constraints.
        """
        self.f = f
        self.prox_f = prox_f
        self.stationarity = stationarity
        self.fc = fc
        self.proj_fc = proj_fc
        self.f_star = f_star
        self.A = A
        self.b = b

    def optimality_gap(self, x):
        """
        Computes the optimality gap of the current solution.

        Parameters:
            x (numpy.ndarray): Primal variable.

        Returns:
            float: Optimality gap of the current solution. 
                    O(x) = max(0, f(x) - f*)
        """
        return np.maximum(self.f(x) - self.f_star, 0)

    def feasibility_gap(self, x):
        """
        Computes the feasibility gap of the current solution.

        Parameters:
            x (numpy.ndarray): Primal variable.

        Returns:
            float: Feasibility gap of the current solution.
                    F(x) = ||Ax - b||
        """
        return np.linalg.norm(self.A @ x - self.b)

    def kkt_error(self, x, y):
        """
        Computes the Karush-Kuhn-Tucker (KKT) error of the current solution.

        Parameters:
            x (numpy.ndarray): Primal variable.
            y (numpy.ndarray): Dual variable.

        Returns:
            float: KKT error of the current solution.
        """
        s = self.stationarity(x, y)
        s = np.linalg.norm(s)**2
        f = self.feasibility_gap(x)**2
        return s + f

    def sdg(self, x, y, beta=np.ones(2)):
        """
        Computes the Smoothed Duality Gap (SDG) of the current solution.

        Parameters:
            x (numpy.ndarray): Primal variable.
            y (numpy.ndarray): Dual variable.
            beta (numpy.ndarray): Smoothing parameter of SDG.

        Returns:
            float: SDG value of the current solution.
        """
        beta_x, beta_y = beta
        p = self.prox_f(x - (1/beta_x)*(self.A.T @ y), 1/beta_x) # Proximal point
        fx, fp = self.f(x), self.f(p)   # f(x) and f(p)
        sdg = fx - fp + (self.A @ (x - p) @ y) - (beta_x/2) * np.linalg.norm(x - p)**2 + (self.feasibility_gap(x))**2 / (2 * beta_y) # SDG
        if sdg <= 0 and sdg > -1e-10:  # Avoiding, potential, numerical issues. 
            sdg = 1e-13
        return sdg

    def pdg(self, x, y):
        """
        Computes the Projected Duality Gap (PDG) of the current solution.

        Parameters:
            x (numpy.ndarray): Primal variable.
            y (numpy.ndarray): Dual variable.

        Returns:
            float: PDG value of the current solution.
        """
        pf = self.feasibility_gap(x)     # Primal feasibility 
        a = self.proj_fc(-1 * self.A.T @ y)  # Projection onto the domain of the Fenchel-Conjugate.
        df = np.linalg.norm(a + (self.A.T @ y))  # Dual feasibility 
        dg = np.abs(self.f(x) + self.fc(a) + np.dot(self.b, y))  # Duality gap
        pdg = pf**2 + df**2 + dg**2  # PDG
        return pdg

