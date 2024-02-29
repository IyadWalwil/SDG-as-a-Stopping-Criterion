import numpy as np 
import pandas as pd
import tqdm.notebook as tn

from OptimizationMeasures import OptimizationMeasures

    
class OptimizationBounds(OptimizationMeasures):
    def __init__(self, primal_var = None, dual_var = None, f = None, prox_f = None, 
                    stationarity = None, fc = None, proj_fc = None, f_star = None, x_star = None, 
                        A = None, b = None, L = None, Lc = None, Lc_grad = None,
                         beta = np.outer(10**(np.linspace(-7, 0, 26)), np.ones(2)), 
                         beta_mode = 'mixed', eta_fun = None, gamma = None):
        """
        Initializes an instance of the OptimizationBounds class.

        Attributes:
            primal_var (list): List of primal variable updates every a pre-defined number of iteration(s).
            dual_var (list): List of dual variable updates every a pre-defined number of iteration(s).
            f (function): Objective function.
            prox_f (function): Proximal operator of the objective function.
            stationarity (function): Stationarity of the Lagrangian function.
            fc (function): Fenchel-Conjugate of the objective function.
            proj_fc (function): Projection onto the domain of fc.
            f_star (float): Optimal value of the objective function.
            x_star (array-like or None): Optimal solution of the optimization problem.
            A (numpy.ndarray): Coefficient matrix of the linear constraints.
            b (numpy.ndarray): Right-hand side vector of the linear constraints.
            L (float): Lipschitz constant of the gradient of the objective function.
            Lc (float): Lipschitz constant of the Fenchel-Conjugate of the objective function.
            Lc_grad (float): Lipschitz constant of the gradient of the Fenchel-Conjugate of the objective function.
            beta (numpy.ndarray): Smoothing parameter of SDG.
                Default: is a geometric array of size (26, 2) such that beta_x = beta_y and beta_x takes 
                          26 equally-divided values in logarithmic scale ranging from 1e-7 to 1. 
                         beta will be re-defined according to beta_mode as, subsequently, explained.
            beta_mode (str): Mode for handling beta by re-defining it appropriately within the whole code.
                Possible values are:
                    * 'cst': beta is constant for all the iterations. 
                        For instance, beta= (1, 1).
                    * 'FG': beta is non-constant and equals the feasibility gap at each iteration. 
                        That is: beta_x = beta_y = ||Ax_k - b|| 
                    * 'mixed' (Default): beta takes a list of constant values in addition to the feasibility gap.
                                         Then, at each iteration k: we choose the beta that minimizes the considered bound. 
            eta_fun (function): Quadratic Error Bound of the Smoothed Gap (QEBSG)-constant, as a function of beta.
            gamma (float): Metric Sub-regularity (MSR)-constant.
        """
        super().__init__(f, prox_f, stationarity, fc, proj_fc, f_star, A, b)
        self.primal_var = primal_var
        self.dual_var = dual_var
        self.x_star = x_star
        self.eta_fun = eta_fun
        self.gamma = gamma
        self.L = L
        self.Lc = Lc
        self.Lc_grad = Lc_grad
        self.beta = beta
        self.beta_mode = beta_mode
        opt_meas = self.optimality_measures() 
        # Computing the OG, FG, KKT and PDG measures. 
        self.OG, self.FG, self.KKT, self.PDG= opt_meas['OG'], opt_meas['FG'], opt_meas['KKT'], opt_meas['PDG']
        # Re-defining beta, and computing the QEBSG-constant, eta. 
        self.beta, self.eta = self.beta_eta_processing(beta)
        # Computing SDG.
        self.SDG = self.SDG_fun(self.beta)
        # Computing the norm of the primal and dual updates at each iteration. 
        self.norm_X, self.norm_Y = self.norm_X(), self.norm_Y()
        # Computing the square root of: KKT, PDG, and SDG.
        self.sq_KKT, self.sq_PDG, self.sq_SDG = np.sqrt(self.KKT), np.sqrt(self.PDG), np.sqrt(self.SDG)

    def beta_eta_processing(self, beta):
        """
        * Re-defining beta appropriately within the whole code based on the selected beta_mode.
        * Computing the corresponding QEBSG-constant, eta.

        Parameter:
            beta (numpy.ndarray): Smoothing parameter of SDG.

        Returns:
            tuple: A tuple containing processed beta and eta values based on the selected beta_mode.
        
        Raises:
            TypeError if the chosen beta_mode is not correct.
        """
        if self.beta_mode == 'cst':  # beta is well-defined.
            return beta, self.eta_fun([beta])
        elif self.beta_mode == 'FG':  # beta equals the feasibility gap.  
            beta_FG = self.FG.copy() 
            beta = np.outer(beta_FG, np.ones(2))  # Assigning beta_x and beta_y the same values
            eta = self.eta_fun(beta)
            return beta.T, eta
        elif self.beta_mode == 'mixed': # beta is a list of constant values and the feasibility gap.
            beta_FG = self.FG.copy()  # Feasibility gap part. 
            eta_FG = self.eta_fun(np.outer(beta_FG, np.ones(2))) # eta for bete equals the feasibility gap
            eta_list = self.eta_fun(beta)  # list of eta values for each constant value of beta
            l1 = len(beta_FG) 
            l2 = len(self.beta.T[0])
            # Re-structuring beta and eta
            BETAx = np.zeros((l1, l2 + 1))  
            BETAy = BETAx.copy()
            ETA = BETAx.copy()
            for i in range(l1):
                BETAx[i, :-1], BETAy[i, :-1] = beta.T[0], beta.T[1]  
                BETAx[i, -1], BETAy[i, -1] = beta_FG[i], beta_FG[i] 
                ETA[i, :-1], ETA[i, -1] = eta_list, eta_FG[i]
            BETA = [BETAx, BETAy]
            return BETA, ETA
        else:
            raise TypeError("The chosen beta_mode is not correct.")

    def optimality_measures(self, measure_dict= {'OG': "Optimality gap", 'FG': "Feasibility gap", 'KKT': "KKT error",  'PDG': "PDG"}):
        """
        Computes the optimality measures (optimality gap, feasibility gap, KKT error, PDG, SDG) at each iteration.

        Parameter:
            measure_list (list): List of strings containing the desired measures to compute.
                                 Default= ['OG', 'FG', 'KKT', 'PDG']

        Returns:
            dist: Dictionary contains the selected optimality measures and their values at each iteration.
        """
        l = len(self.primal_var)
        measures = {
            'OG': (np.zeros(l), lambda x, y: self.optimality_gap(x)),
            'FG': (np.zeros(l), lambda x, y: self.feasibility_gap(x)),
            'KKT': (np.zeros(l), lambda x, y: self.kkt_error(x, y)),
            'PDG': (np.zeros(l), lambda x, y: self.pdg(x, y)),
        }

        selected_measures = [measures[measure] for measure in measure_dict.keys()]
        for measure_name, (measure_array, measure_function) in zip(measure_dict.items(), selected_measures):
            inner_bar = tn.trange(len(self.primal_var), desc=measure_name[1])
            for (k, x, y) in zip(inner_bar, self.primal_var, self.dual_var):
                measure_array[k] = measure_function(x, y)
                inner_bar.set_postfix(**{measure_name[0]: measure_array[k]})

        return {key: measure_array for key, (measure_array, _) in zip(measure_dict.keys(), selected_measures)}
    
    def SDG_fun(self, BETA):
        """
        Computes the Smoothed Dual Gap (SDG) at each iteration based on the selected beta_mode.
            - if beta_mode is cst: it computes SDG at each iteration using the same constant beta.
            - if beta_mode is FG: it computes SDG at each iteration with beta = ||Ax_k - b||. 
            - if beta_mode is mixed: it computes SDG at each iteration for each beta. 
            
        Parameter:
            beta (numpy.ndarray): Smoothing parameter of SDG.

        Returns:
            numpy.ndarray: An array containing the Smoothed Dual Gap values at each iteration.
        
        Raises: 
            TypeError if the chosen beta_mode is not correct. 
        """
        l = len(self.primal_var)
        bar = tn.tnrange(l)
        if self.beta_mode == 'cst':
            SDG = np.zeros(l)
            for (k, x, y) in zip(bar, self.primal_var, self.dual_var):
                SDG[k] = self.sdg(x, y, BETA)
                bar.set_postfix(SDG=SDG[k])
                bar.set_description("SDG with constant beta")
        elif self.beta_mode == 'FG':
            SDG = np.zeros(l)
            for (k, x, y) in zip(bar, self.primal_var, self.dual_var):
                SDG[k] = self.sdg(x, y, BETA.T[k])
                bar.set_postfix(SDG=SDG[k])
                bar.set_description("SDG with beta equals FG")
        elif self.beta_mode == 'mixed':
            SDG = np.zeros_like(BETA[0])
            BETAx, BETAy = BETA[0], BETA[1]
            for (k, x, y) in zip(bar, self.primal_var, self.dual_var):
                for i in range(SDG.shape[1]):
                    SDG[k, i] = self.sdg(x, y, (BETAx[k, i], BETAy[k, i]))
                    bar.set_postfix(SDG=SDG[k, i])
                    bar.set_description("SDG matrix with multiple beta at each iteration")
        else: 
            raise TypeError("The chosen beta_mode is not correct.")
        return SDG

    def norm_X(self):
        """
        Computes the norm of the primal variable at each iteration.

        Returns:
            numpy.ndarray: The computed norm of the primal variable.
        """
        return np.linalg.norm(self.primal_var, axis = 1)
        
    def norm_Y(self):
        """
        Computes the norm of the dual variable at each iteration.

        Returns:
            numpy.ndarray: The computed norm of the dual variable.
        """
        return np.linalg.norm(self.dual_var, axis = 1)

########### Optimality Gap Bounds ###########
# Result: KKT approximation for the optimality gap assuming MSR.
    def OG_KKT(self):
        """
        Computes the KKT approximation for the optimality gap.
            Approx. = (2/gamma)*KKT + ||y||*sqrt(KKT)

        Returns:
            numpy.ndarray: Array of the computed approximation.
        """
        return (2 / self.gamma) * self.KKT + (self.norm_Y * self.sq_KKT)

# Result: SDG approximation for the optimality gap assuming QEBSG.
    def OG_SDG(self):
        """
        Computes the SDG approximation for the optimality gap.
            Approx. = (1 + 2 sqrt(beta_x/eta))*SDG + sqrt(2*beta_y)*||y||*sqrt(SDG)

        Returns:
            numpy.ndarray: Array of the computed approximation.
        """
        if self.beta_mode == "mixed":
            BETAx, BETAy = self.beta.copy()
            ETA = self.eta.copy()
            sq_beta_y = np.sqrt(2 * BETAy)
            beta_eta = 1 + (2 * np.sqrt((BETAx/ETA)))
            bound = (beta_eta * self.SDG) + (self.norm_Y.reshape(-1, 1) * sq_beta_y  * np.sqrt(self.SDG))
            bound = np.min(bound, axis=1)  # Finding the minimum over beta at each iteration.
            return bound
        else:
            beta_x, beta_y = self.beta.copy()
            eta = self.eta.copy()
            sq_beta_y = np.sqrt(2 * beta_y)
            beta_eta = 1 + (2 * np.sqrt((beta_x/eta)))
            bound = (beta_eta * self.SDG) + (self.norm_Y * sq_beta_y  * np.sqrt(self.SDG))
            return bound
    
# Result: PDG approximation for the optimality gap assuming QEBSG.
    def OG_PDG(self):
        """
        Computes the PDG approximation for the optimality gap.
            Approx. = (1 + ||x|| + sqrt(2/eta) * sqrt((1 + ||x|| + ||y||)*sqrt(PDG) + (1/(2*beta_min))*PDG))*sqrt(PDG)
                        beta_min = min(beta_x, beta_y)

        Returns:
            numpy.ndarray: Array of the computed approximation.
        """
        if self.beta_mode == 'mixed':
            BETAx, BETAy = self.beta.copy()
            ETA = self.eta.copy()
            beta_bar = np.minimum(BETAx, BETAy)
            SDG_bound = ((1 + self.norm_X + self.norm_Y) * self.sq_PDG).reshape(-1, 1) + ((1/ (2 * beta_bar)) * self.PDG.reshape(-1, 1))
            bound = (1 + self.norm_X.reshape(-1, 1) + np.sqrt((2*SDG_bound)/ETA))*self.sq_PDG.reshape(-1, 1)
            bound = np.min(bound, axis=1) 
            return bound
        else:
            SDG_PDG = self.SDG_PDG()[0].reshape(-1)
            return (1 + self.norm_X + np.sqrt(SDG_PDG/self.eta))*self.sq_PDG

# Optimality gap bounds together.
    def OG_bounds(self):
        """
        Gathers the optimality gap bounds.

        Returns:
            dict: Dictionary containing the optimality gap bounds: KKT, SDG, and PDG, respectively.
        """
        print("""
╔══════════════════════════════════════╗
║        Optimality Gap Bounds         ║
╚══════════════════════════════════════╝""")
        print("")
        KKT_bound = self.OG_KKT()
        print("* KKT approximation: Done")
        SDG_bound = self.OG_SDG()
        print("* SDG approximation: Done")
        PDG_bound = self.OG_PDG()
        print("* PDG approximation: Done")
        return {'KKT': KKT_bound, 'SDG': SDG_bound, 'PDG': PDG_bound}

########### Comparability Bounds ###########
# Result: SDG approximation for KKT assuming that the objective function is differentiable and has an L-Lipschitz gradient.             
    def KKT_SDG(self):
        """
        Computes the SDG approximation for the KKT error.
            Approx. = β̲ * KKT   with β̲ = max(1/beta_x, 1/(2*beta_y))  

        Returns:
            numpy.ndarray: Array of the computed approximation. 
        """
        if self.beta_mode == "mixed":
            BETAx, BETAy = self.beta.copy()
            beta_L = np.maximum((2 * (self.L + BETAx)**2) / BETAx , 2 * BETAy)  
            bound =  beta_L * self.SDG      
            return np.min(bound, axis=1) # Finding the minimum over beta at each iteration.
        else:
            beta_x, beta_y = self.beta.copy()
            beta_L = np.maximum((2 * (self.L + beta_x)**2) / beta_x , 2 * beta_y)        
            return beta_L * self.SDG  # Bound.


# Result: KKT approximation for the smoothed duality gap.
    def SDG_KKT(self):
        """
        Computes the KKT approximation for SDG.
            Approx. = beta_L * SDG   with beta_L = max((2*(L + beta_x)^2)/(beta_x), 2*beta_y)

        Returns:
            dict: A dictionary containing two arrays:
                  'bound': Array of computed upper bound based on the KKT error.
                  'SDG': Array of SDG values. 
        """
        if self.beta_mode == "mixed":
            SDG_copy = self.SDG.copy()
            BETAx, BETAy = self.beta.copy()
            beta_lbar = np.maximum(1 / BETAx, 1 / (2 * BETAy))
            l = len(self.KKT)
            bound = beta_lbar * self.KKT.reshape(l, 1) - self.SDG
            min_indices = np.argmin(bound, axis=1)  # Finding the index of the minimal beta at each iteration.
            beta_min = beta_lbar[np.arange(l), min_indices]  # Minimal beta at each iteration.
            SDG_min = SDG_copy[np.arange(l), min_indices]   # Minimal SDG over beta at each iteration.
            return {'bound': beta_min * self.KKT, 'SDG': SDG_min}
        else:
            beta_x, beta_y = self.beta.copy()
            beta_lbar = np.maximum(1 / beta_x, 1 / (2 * beta_y))
            res = beta_lbar * self.KKT 
            return {'bound': res, 'SDG': self.SDG}
    
# KKT and SDG bounds together 
    def KKT_SDG_bounds(self):
        """
        Gathers the KKT and SDG bounds.

        Returns:
            dict: A dictionary containing two arrays:
                  'K<G': Array of the SDG approximation for the KKT error.
                  'G<K': Array of the KKT approximation for SDG.
        """
        print("""
╔══════════════════════════════════════╗
║        Comparability Bounds          ║
╚══════════════════════════════════════╝""")
        KG = self.KKT_SDG()
        print("* SDG approximation for KKT: Done")
        GK = self.SDG_KKT()
        print("* KKT approximation for SDG: Done")
        return {'K<G': KG, 'G<K': GK}
    
# Result: PDG approximation for the smoothed duality gap.
    def SDG_PDG(self):
        """
        Computes the PDG approximation for SDG.

        Returns:
            dict: A dictionary containing two arrays:
                  'bound': Array of the computed upper bound based on PDG.
                  'SDG': Array of SDG values.
        """
        if self.beta_mode == "mixed":
            SDG_copy = self.SDG.copy()
            BETAx, BETAy = self.beta.copy()
            beta_bar = np.minimum(BETAx, BETAy)
            term1 = (1 + self.norm_X + self.norm_Y) * self.sq_PDG
            term2 = ((1/ (2 * beta_bar)) * self.PDG.reshape(-1, 1))
            bound_ratio = (term1.reshape(-1, 1)/SDG_copy) + (term2/SDG_copy)
            min_indices_ratio = np.argmin(bound_ratio, axis=1)   # Finding the index of the minimal beta at each iteration.
            l = len(self.PDG)
            beta_min_ratio = beta_bar[np.arange(l), min_indices_ratio] # Minimal beta at each iteration.
            SDG_min_ratio = SDG_copy[np.arange(l), min_indices_ratio]  # Minimal SDG over beta at each iteration 
            bound = term1.reshape(-1) + ((1/(2*beta_min_ratio)) * self.PDG)
            return {'bound': bound, 'SDG': SDG_min_ratio}
        else:
            beta_x, beta_y = self.beta.copy()
            beta_bar = np.minimum(beta_x, beta_y)
            bound = ((1 + self.norm_X + self.norm_Y) * self.sq_PDG) + ((1/ (2 * beta_bar)) * self.PDG)
            return {'bound': bound, 'SDG': self.SDG}
    
# Result: SDG approximation for the projected duality gap.
    def PDG_SDG(self):
        """
        Computes the SDG approximation for PDG.
            Approx. = β̲ * KKT   with β̲ = max(1/beta_x, 1/(2*beta_y))  

        Returns:
            numpy.ndarray: Array of the computed approximation. 
        """
        if self.beta_mode == "mixed":
            BETAx, BETAy = self.beta.copy()
            beta_Bar = np.maximum(BETAx, BETAy)
            term1 = 2 * beta_Bar * self.SDG
            if self.Lc is not None: # If the Fenchel-Conjugate is L-Lipschitz. 
                term2 = np.sqrt(2) * ((np.sqrt(BETAx) * (self.norm_X.reshape(-1, 1) + self.Lc)) + (np.sqrt(BETAy) * self.norm_Y.reshape(-1, 1))) * self.sq_SDG
                bound = term1 + (self.SDG + term2)**2
                return np.min(bound, axis=1)  # Finding the minimum over beta at each iteration.
            else:   # If the gradient of the Fenchel-Conjugate is L-Lipschitz.
                term2 = np.sqrt(2) * ((2 * np.sqrt(BETAx) * self.norm_X.reshape(-1, 1)) + (np.sqrt(BETAy) * self.norm_Y.reshape(-1, 1))) * self.sq_SDG
                bound = term1 + (((3 + (BETAx * self.Lc_grad)) * self.SDG) + term2)**2
                return np.min(bound, axis=1)  # Finding the minimum over beta at each iteration.
        else:
            beta_x, beta_y = self.beta.copy()
            beta_Bar = np.maximum(beta_x, beta_y)
            term1 = 2 * beta_Bar * self.SDG
            if self.Lc is not None:  # If the Fenchel-Conjugate is L-Lipschitz. 
                print('PDG-SDG: Lipschitz function')
                term2 = np.sqrt(2) * ((np.sqrt(beta_x) * (self.norm_X + self.Lc)) + (np.sqrt(beta_y) * self.norm_Y)) * self.sq_SDG
                bound = term1 + (self.SDG + term2)**2
                return bound
            else:  # If the gradient of the Fenchel-Conjugate is L-Lipschitz.
                print('PDG-SDG: Lipschitz gradient')
                term2 = np.sqrt(2) * ((np.sqrt(beta_x) * self.norm_X) + (np.sqrt(beta_y) * self.norm_Y)) * self.sq_SDG
                bound = term1 + (((3 + (beta_x * self.Lc_grad)) * self.SDG) + term2)**2
                return bound 
        
# PDG and SDG bounds together 
    def PDG_SDG_bounds(self):
        """
        Gathers the PDG and SDG bounds.

        Returns:
            dict: A dictionary containing two arrays:
                  'G<D': Array of the PDG approximation for SDG.
                  'D<G': Array of the SDG approximation for PDG.
        """
        print("* PDG approximation for SDG: Done")
        GD = self.SDG_PDG()
        print("* SDG approximation for PDG: Done")
        DG = self.PDG_SDG()
        return {'G<D': GD, 'D<G': DG}
    
# Feasibility Gap Results Together 
    def FG_results(self):
        """
        Computes the feasibility gap results together.

        Returns:
            A dictionary containing arrays of the computed feasibility gap results.
        """

        sq_beta_y = np.sqrt(2*self.beta)
        FK = self.sq_KKT
        if self.beta_mode == 'mixed':
            FG = np.min(sq_beta_y * self.sq_SDG, axis = 1)
        else:
            FG = sq_beta_y * self.sq_SDG
        FD = self.sq_PDG
        return {'KKT': FK, 'SDG': FG, 'PDG': FD}

# All the bounds 
    def all_bounds(self):
        """
        Gathers all the bounds.

        Returns:
            tuple: A tuple containing: the optimality gap bounds, and the comparability bounds (KKT-SDG & PDG-SDG).
        """
        return self.OG_bounds(), self.KKT_SDG_bounds(), self.PDG_SDG_bounds()
    
def find_iter(problems_bounds, eps, columns=['KKT error', 'SDG', 'PDG']):
    """
    Finds the iteration, k, at which the bounds, the optimality gap approximation in our case, for each problem fall below a given epsilon.

    Parameters:
        problems_bounds (dict): A dictionary where keys are problem identifiers and values are tuples 
                                containing a dictionary of bounds for each problem and the iteration step size.
        eps (float): The epsilon value to compare the bounds against.
        columns (list): A list of column names for the resulting DataFrame.
                        Default: ['KKT error', 'SDG', 'PDG']

    Returns:
        pandas.DataFrame: A DataFrame where rows correspond to problems and columns correspond to the considered bounds,
                          with values representing the iteration at which each bound falls below the epsilon.
    """
    res = {}
    for prob, (bounds, step) in problems_bounds.items():
        indices = []
        for bound in bounds.values():
            index = np.where(bound < eps)[0]
            if len(index) > 0:
                indices.append(index[0]*step)
            else:
                indices.append(np.inf)
        res[prob] = indices
    df = pd.DataFrame.from_dict(res, orient='index', columns=columns)
    return df

def avg_sd(problems_bounds):
    """
    Computes the average and standard deviation of the ratio between the second and first elements 
    of each bound array for multiple problems. 
    For instance, for a result: M <= W(N), the bound array is (M, W(N)) and hence the ratio is: W(N)/M.

    Parameters:
        problems_bounds (dict): A dictionary where keys are problem identifiers and values are dictionaries 
                                containing different bounds for each problem.

    Returns:
        pandas.DataFrame: A DataFrame where rows correspond to problems and columns correspond to the considered bounds,
                          with values representing the average and standard deviation of the ratio between 
                          the second and first elements of each bound array.
    """
    res = {}
    columns = list(next(iter(problems_bounds.values())).keys())
    for prob, bounds in problems_bounds.items():
        values = []
        for bound in bounds.values():
            ratio = bound[1]/bound[0]
            avg, sd = np.mean(ratio), np.std(ratio)
            if avg > 999 and avg != np.inf:
                avg = exponent_format(avg)
            else: 
                avg = round(avg, 2)
            if sd > 999 and sd is not None:
                sd = exponent_format(sd)
            else:
                sd = round(sd, 2)
            values.append(str(avg) + " ± " + str(sd))
        res[prob] = values
    df = pd.DataFrame.from_dict(res, orient='index', columns=columns)
    return df

def exponent_format(num):
    temp = "{:e}".format(num)
    new, exp = temp.split('e+')
    res = str(round(float(new), 2)) + 'e+' + exp
    return res