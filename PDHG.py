import numpy as np
from OptimizationMeasures import OptimizationMeasures
from tqdm import tqdm as tqdm
import tqdm.notebook as tn
import scipy.sparse as sp


class PDHG:
    """
    - Primal-Dual Hybrid Gradient (PDHG) algorithm for solving optimization problems of the form:

            min_x max_y f(x) + <Ax - b, y> 

        where f(x) is a convex, lower semi-continuos, and proper function.

    - PDHG version 1
    """

    def __init__(self, xo, yo, prox_f, A, b, x_star=None, tau=None, sigma=None):
        """
        Initialize the PDHG algorithm with given parameters.

        Attributes:
            xo (numpy.ndarray): Initial value of the primal variable.
            yo (numpy.ndarray): Initial value of the dual variable.
            prox_f (function): Proximal operator of the objective function, f(x).
            A (numpy.ndarray or scipy.sparse.coo_matrix): Coefficient matrix of the linear constraints.
            b (numpy.ndarray): Right-hand side vector of the linear constraints.
            x_star (numpy.ndarray, optional): Optimal solution.
            tau (float, optional): Primal step size.
            sigma (float, optional): Dual step size.
        """
        self.xo = xo
        self.yo = yo
        self.prox_f = prox_f
        self.A = A
        self.b = b
        self.tau, self.sigma = tau, sigma
        self.tau, self.sigma = self.step_sizes()
        self.x_star = x_star

    def step_sizes(self):
        """
        Calculate step sizes tau and sigma if not provided.

        Returns:
            tuple: Primal-dual step sizes (tau, sigma).
        """
        if self.tau is None:
            if isinstance(self.A, sp.coo_matrix):
                normA = sp.linalg.norm(self.A, ord=2)
            ###
            else:
                normA = np.linalg.norm(self.A, 2)
            self.tau = 0.95 / normA
            self.sigma = 1 / normA
        ###
        return self.tau, self.sigma

    def stopping_criterion(self, x, y, stop_crt, f=None, stationarity=None, beta=None, eta_fun=None, gamma=None):
        """
        Compute the gap of the employed stopping criterion.

        Parameters:
            x (numpy.ndarray): Current primal variable.
            y (numpy.ndarray): Current dual variable.
            stop_crt (str): Employed stopping criterion.
                * 'Dtto': Distance to the optimum. 
                * 'KKT error': Karush–Kuhn–Tucker error. 
                * 'SDG': Smoothed Duality Gap.
            f (function): Objective function.
            stationarity (function): Stationarity of the Lagrangian function.
            beta (float): Smoothing parameter of SDG.
            eta_fun (function): Quadratic Error Bound of the Smoothed Gap (QEBSG)-constant, as a function of beta.
            gamma (float): Metric Sub-regularity (MSR)-constant.

        Returns:
            float: Gap of the employed stopping criterion. 
        """
        if stop_crt == 'DttO':
            if self.x_star is None:
                raise Exception("You have to provide x_star to use DttO")
            # Compute the stopping criterion based on the distance to the optimum.
            gap = np.linalg.norm(x - self.x_star)
        ###
        elif stop_crt == 'KKT error':
            """
            Compute the stopping criterion based on the KKT error approximation for the optimality gap (Theorem 2).
                Approx. = (2/gamma)*KKT + ||y||*sqrt(KKT)
            """            
            obj = OptimizationMeasures(f=f, stationarity=stationarity, A=self.A, b=self.b)  # Create an instance of the OptimizationMeasures class.
            kkt = obj.kkt_error(x, y)   # KKT error.
            norm_y = np.linalg.norm(y)  # Norm of the dual variable y.
            gap = (2 / gamma) * kkt + (norm_y * np.sqrt(kkt))  # KKT approximation.  
        ###
        elif stop_crt == 'SDG':
            """
            Compute the stopping criterion based on the smoothed duality gap approximation for the optimality gap (Theorem 3).
                Approx. = (1 + 2 sqrt(beta_x/eta))*SDG + sqrt(2*beta_y)*||y||*sqrt(SDG)
            """
            obj = OptimizationMeasures(f=f, prox_f=self.prox_f, A=self.A, b=self.b) # Create an instance of the OptimizationMeasures class.
            if beta is None: 
                beta = obj.feasibility_gap(x)  # Initialize the parameter 'beta' if it has not been initialized already.
                beta = np.ones(2)*beta
            beta_x, beta_y = beta
            sdg = obj.sdg(x, y, beta)  # SDG.
            norm_y = np.linalg.norm(y) # Norm of the dual variable y.
            sq_beta_y = np.sqrt(2 * beta_y) 
            eta = eta_fun(beta)   # QEBSG-constant. 
            beta_eta = 1 + (2 * np.sqrt((beta_x / eta)))
            sq_sdg = np.sqrt(sdg)  # square root of SDG
            gap = (beta_eta * sdg) + (sq_beta_y * norm_y * sq_sdg)  # SDG approximation. 
        ###
        else: # In case we chose another stopping criterion. 
            raise TypeError('''The employed stopping criterion is not correct, it should be one of the following:
                                * 'DttO': Distance to the Optimum.
                                * 'KKT error':  Karush-Kuhn-Tucker error
                                * 'SDG': Smoothed Duality Gap''') 
        ###
        return gap

    def update_variables(self, x, y):
        """
        Update the primal and dual variables based on the PDHG algorithm.

        Parameters:
            x (numpy.ndarray): Current primal variable.
            y (numpy.ndarray): Current dual variable.

        Returns:
            tuple: Updated primal and dual variables (x, y).
        """
        # Primal Forward-Backward step.
        u = x - self.tau * (self.A.T @ y)  
        x = self.prox_f(u, self.tau)
        # Dual Forward-Backward step. 
        y_bar = y + self.sigma * (self.A @ x - self.b)
        # Primal Extrapolation step.
        x -= self.tau * (self.A.T @ (y_bar - y))
        # Dual Extrapolation step.
        y = y_bar
        return x, y

    def solve(self, stop_crt='SDG', tol=1e-9, stop_crt_kwargs={}, first_run_iter=None):
        """
        Solve the optimization problem using the PDHG algorithm.

        Parameters:
            stop_crt (str): Stopping criterion to employ.
                Default: smoothed duality gap.
            tol (float): Tolerance for the stopping criterion.
                Default: 1e-8
            stop_crt_kwargs (dict): Additional keyword arguments for the employed stopping criterion.
            first_run_iter (int): Number of iterations required by the first run.

        Returns:
            if 'DttO', 'KKT error', or 'SDG' has been employed: 
                dict: Dictionary containing
                    * Number of iterations
                    * Primal optimal solution.
                    * Dual optimal solution.
            if 'FNoI' has been employed: 
                dict: Dictionary containing
                    * Step: it means that we append the primal-dual solutions to primal-dual lists every step iteration. 
                    * Primal_var: list of the appended primal variables. 
                    * Dual_var: list of the appended dual variables. 

        """
        x_prev, y = self.xo, self.yo  # Primal-dual initialization 
        if stop_crt != "FNoI":  # If the employed stopping criterion is one of: 'DttO', 'KKT error', or 'SDG'.
            print('Stopping Criterion:', stop_crt)

            # Stopping variables initialization.
            counter = 0   
            gap = np.inf 
            # Perform PDHG iterations until the stopping criterion is met or the maximum number of iterations is reached
            print('Progress:') 
            progress = tqdm() 
            while (gap >= tol and counter <= int(1e6)): # Stopping condition 
                counter += 1  
                x, y = self.update_variables(x_prev, y) # Primal-dual update
                gap = self.stopping_criterion(x, y, stop_crt, **stop_crt_kwargs)
                progress.update()
                progress.set_postfix(Gap=gap)
                x_prev = x
            return {"No. of iterations": counter, "Primal optimal": x, "Dual optimal": y}
        else:
            print('Stopping criterion: Fixed number of iterations =', first_run_iter)
            step = np.maximum(1, first_run_iter // 1500)
            print('We append the primal and dual solutions every', step, 'iteration(s)')
            primal_var, dual_var = [x_prev], [y]  # Initializing the primal-dual lists with the primal-dual initialization.

            # Perform a fixed number of PDHG iterations
            print("Progress:")
            bar = tn.trange(first_run_iter)
            for i in bar:
                x, y = self.update_variables(x_prev, y)  # Primal-dual update
                x_prev = x
                if i % step == 0:  # Every 'step' iteration, we append 
                    primal_var.append(x)  # the primal variable 
                    dual_var.append(y)    # the dual variable 
            if first_run_iter % step != 0: # We append the primal-dual optimal solutions if they have not been appended already.
                primal_var.append(x)
                dual_var.append(y)
            return {"step": step, "Primal variables": primal_var, "Dual variables": dual_var}

    def two_runs(self, stop_crt='SDG', tol=1e-9, stop_crt_kwargs={}):
        """
        Perform two runs of the PDHG algorithm. 
            - The first run is done only to identify the required number of iterations.
            - In the second run, we store around 1500 primal and dual updates evenly-distributed. 

        Parameters:
            stop_crt (str): Stopping criterion to employ.
            tol (float): Tolerance for the stopping criterion.
            stop_crt_kwargs (dict): Additional keyword arguments for the employed stopping criterion.

        Returns:
            dict: Dictionary containing the results of the second run.
                As in the 'solve' function: {"step": step, "Primal variables": primal_var, "Dual variables": dual_var}
        """
        print("""
╔══════════════════════════════════════╗
║              FIRST RUN               ║
╚══════════════════════════════════════╝""")
        first_run = self.solve(stop_crt, tol, stop_crt_kwargs) 
        print("""
╔══════════════════════════════════════╗
║             SECOND RUN               ║
╚══════════════════════════════════════╝""")
        second_run = self.solve(stop_crt='FNoI', first_run_iter=first_run['No. of iterations'])
        return second_run

class PDHG2(PDHG):
    """
    PDHG, version 2, with different variable update order.
    """
    def __init__(self, xo, yo, prox_f, A, b, x_star=None, tau=None, sigma=None):
        """
        Initialize PDHG2.

        Parameters are the same as PDHG.
        """
        super().__init__(xo, yo, prox_f, A, b, x_star, tau, sigma)

    def update_variables(self, x, y):
        """
        Update the primal and dual variables based on the second version of PDHG.

        Parameters:
            x (numpy.ndarray): Current primal variable.
            y (numpy.ndarray): Current dual variable.

        Returns:
            tuple: Updated primal and dual variables (x, y).
        """
        # Dual Forward-Backward step
        y_bar = y + self.sigma * (self.A @ x - self.b)
        # Primal Forward-Backward step
        u = x - self.tau * (self.A.T @ y_bar)
        x_bar = self.prox_f(u, self.tau)
        # Dual Extrapolation step
        y = y_bar + self.sigma * (self.A @ (x_bar - x))
        # Primal Extrapolation step
        x = x_bar
        return x, y
    
class PDHG3(PDHG): 
    """
    In this sub-class, we update the primal and dual variables the same as the PDHG class. 
    The only difference is that we return x_bar in addition to x. 
    Then, we compute the measures and the bounds at x_bar instead of x. 
    """
    def __init__(self, xo, yo, prox_f, A, b, x_star=None, tau = None, sigma = None):
        super().__init__(xo, yo, prox_f, A, b, x_star, tau, sigma)

    def update_variables(self, x_prev, y):
        # Update the primal and dual variables based on the PDHG algorithm steps
        u = x_prev - self.tau * (self.A.T @ y)
        x_bar = self.prox_f(u, self.tau)
        y_bar = y + self.sigma * (self.A @ x_bar - self.b)
        x = x_bar - self.tau * (self.A.T @ (y_bar - y))
        y = y_bar
        return x_bar, x, y

    def solve(self, stop_crt='SDG', tol=1e-8, stop_crt_kwargs={}, first_run_iter=None):
        x_prev, y = self.xo, self.yo  # Primal-dual initialization 
        if stop_crt != "FNoI":  # If the employed stopping criterion is one of: 'DttO', 'KKT error', or 'SDG'.
            print('Stopping Criterion:', stop_crt)

            # Stopping variables initialization.
            counter = 0   
            gap = np.inf 
            # Perform PDHG iterations until the stopping criterion is met or the maximum number of iterations is reached
            print('Progress:') 
            progress = tqdm() 
            while (gap >= tol and counter <= int(1e6)): # Stopping condition 
                counter += 1  
                x_bar, x, y = self.update_variables(x_prev, y) # Primal-dual update
                gap = self.stopping_criterion(x_bar, y, stop_crt, **stop_crt_kwargs)
                progress.update()
                progress.set_postfix(Gap=gap)
                x_prev = x
            return {"No. of iterations": counter, "Primal optimal": x, "Dual optimal": y}
        else:
            print('Stopping criterion: Fixed number of iterations =', first_run_iter)
            step = np.maximum(1, first_run_iter // 1500)   
            print('We append the primal and dual solutions every', step, 'iteration(s)')
            primal_var, primal_bar_var, dual_var = [x_prev], [x_prev], [y]

            # Perform a fixed number of PDHG iterations
            print("Progress:")
            bar = tn.trange(first_run_iter)
            for i in bar:
                x_bar, x, y = self.update_variables(x_prev, y)  # Primal-dual update
                x_prev = x
                if i % step == 0:  # Every 'step' iteration, we append 
                    primal_var.append(x)  # the primal variable 
                    dual_var.append(y)    # the dual variable 
                    primal_bar_var.append(x_bar)
            if first_run_iter % step != 0: # We append the primal-dual optimal solutions if they have not been appended already.
                primal_var.append(x)
                dual_var.append(y)
                primal_bar_var.append(x_bar)
            return {"step": step, "Primal variables": primal_var, "Primal bar variables": primal_bar_var, "Dual variables": dual_var}
