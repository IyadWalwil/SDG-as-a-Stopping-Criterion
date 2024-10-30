import numpy as np
import scipy.sparse as sp
import cvxpy as cp 
from matplotlib import pyplot as plt

class LeastSquares:
    """
    Class encapsulating all the necessary analysis of the Least-Squares problem, as presented is Subsection 7.1 in the article, for:
        - Solving the optimization problem: min ||Qx - c||^2 s.t. Ax = b. 
        - Computing the different measures we consider. 
        - Computing our findings bounds. 

    Methods:
        - __init__: Initializes the Least-Squares problem with the given attributes.
        - LS: Objective function.
        - stationarity: Computes the gradient of the Lagrangian w.r.t. x.
        - prox_LS: Proximal operator of the objective function.
        - LSc: Fenchel-Conjugate of the objective function.
        - proj_LSc: Projection onto the domain of the Fenchel-Conjugate function.
        - grad_Lipschitz: Computes the Lipschitz constant of the gradient of the objective function.
        - gradc_Lipschitz: Computes the Lipschitz constant of the gradient of the Fenchel-Conjugate function.
        - matrix_H: Hessian of SDG for least-squares.
        - matrix_M: Metric Sub-Regularity (MSR) matrix.
        - eta_QEBSG: Computes the Quadratic Error Bound of the Smoothed Gap (QEBSG) constant eta.
        - ETA_BETAs: Computes QEBSG constants ETA for several betas.
        - gamma_MSR: Computes the MSR constant gamma.
        - solve: Solves the least squares problem using CVXPY.
        - prob_parameters: Computes problem parameters such as Lipschitz constants, MSR constant, optimal solution, etc...
    """
    def __init__(self, n, m, Q, c, A, b, beta = None):
        """
        Initializes the least squares problem with the given attributes.

        Attributes:
            n (int): Dimension of the decision variable.
            m (int): Number of constraints.
            Q (numpy.ndarray): Coefficient matrix Q in the least squares objective.
            c (numpy.ndarray): Vector c in the least squares objective.
            A (numpy.ndarray): Coefficient matrix of the linear constraints.
            b (numpy.ndarray): Right-hand side vector of the constraints.
            beta (numpy.ndarray): Smoothing parameter of SDG.
        """
        self.n = n
        self.m = m
        self.Q = Q
        self.c = c
        self.A = A
        self.b = b
        self.beta = beta
        self.QQ_inv = np.linalg.pinv(self.Q.T @ self.Q) # Pseudo-inverse of Q.T@Q
        # Checking whether there is a solution. 
        if isinstance(self.A, sp.coo_matrix):
            check_sol = sp.linalg.lsqr(A, b)[4]
        ###
        else:
            check_sol = np.linalg.norm(A @ np.linalg.lstsq(A, b, rcond=None)[0] - b)
        print('Solution Existence:', check_sol < 1e-12, '\t', '|Ax - b| = ', check_sol)

    def LS(self, x):
        """
        Computes the value of the objective function.

        Parameter:
            x (numpy.ndarray): Decision variable.

        Returns:
            float: Value of the objective function.
        """
        res = self.Q @ x - self.c
        res = 0.5*np.linalg.norm(res)**2
        return res

    def stationarity(self, x, y):
        """
        Computes the gradient of the Lagrangian w.r.t. x (Stationarity).

        Parameters:
            x (numpy.ndarray): Decision variable.
            y (numpy.ndarray): Lagrange multiplier (dual variable).

        Returns:
            numpy.ndarray: Gradient of the Lagrangian w.r.t. x.
        """
        grad_LS = self.Q.T @ (self.Q @ x - self.c)
        res = grad_LS + self.A.T @ y
        return res

    def prox_LS(self, x, s):
        """
        Computes the proximal of the objective function.

        Parameters:
            x (numpy.ndarray): Decision vector.
            s (float): Step size.

        Returns:
            numpy.ndarray: Proximal of the objective function.
        """
        M = self.Q.T @ self.Q + (1/s)*np.identity(self.Q.shape[1])
        v = self.Q.T @ self.c + (1/s)*x 
        return np.linalg.solve(M, v)

    def LSc(self, mu):
        """
        Computes the Fenchel-Conjugate of the objective function.

        Parameter:
            mu (numpy.ndarray): Fenchel-Conjugate variable.

        Returns:
            float: Value of the Fenchel-Conjugate.
        """
        x_bar = self.QQ_inv @ ((self.Q.T @ self.c) + mu)
        res = np.dot(mu, x_bar) - (0.5 * np.linalg.norm((self.Q @ x_bar) - self.c)**2) 
        return res 

    def proj_LSc(self, v):
        """
        Projects onto the domain of the Fenchel-Conjugate function.

        Parameter:
            v (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Projected vector.
        """
        xo = np.linalg.pinv(self.Q @ self.Q.T) @ self.Q @ v
        return (self.Q.T @ xo)

    def grad_Lipschitz(self):
        """
        Computes the Lipschitz constant of the gradient of the objective function.

        Returns:
            float: Lipschitz constant.
        """
        return np.linalg.norm(self.Q.T @ self.Q, 2)

    def gradc_Lipschitz(self):
        """
        Computes the Lipschitz constant of the gradient of the Fenchel-Conjugate function.

        Returns:
            float: Lipschitz constant.
        """
        return np.linalg.norm(self.QQ_inv, 2)
    
    def matrix_H(self, beta): 
        """
        Hessian matrix of SDG for least-squares.

        Parameter:
            beta (numpy.ndarray): Smoothing parameter of SDG.

        Returns:
            numpy.ndarray: Hessian matrix of SDG.
        """
        beta_x, beta_y = beta
        B = self.Q.T @ self.Q + beta_x*np.identity(self.Q.shape[1]) 
        B_inv = np.linalg.inv(B)
        Mx = self.Q.T @ self.Q + (1/(beta_y + 1e-18))*(self.A.T @ self.A) + (beta_x**2)*B_inv - beta_x*np.identity(self.Q.shape[1])
        My = self.A @ B_inv @ self.A.T
        Mxy = self.A.T - beta_x * (B_inv @ self.A.T)
        H = np.block([[Mx,Mxy],[Mxy.T,My]])
        # print("H = ", H)
        return 0.5*H

    def matrix_M(self):
        """
        Computes the MSR matrix.

        Returns:
            numpy.ndarray: MSR matrix.
        """
        return np.block([[self.Q.T @ self.Q, self.A.T], [self.A, np.zeros((self.A.shape[0], self.A.shape[0]))]])

    def eta_QEBSG(self, beta):
        """
        Computes the Quadratic Error Bound of the Smoothed Gap (QEBSG) constant eta.

        Parameter:
            beta (numpy.ndarray): Smoothing parameter of SDG.

        Returns:
            float: QEBSG constant eta.
        """
        eig_values = np.linalg.eig(self.matrix_H(beta))[0] 
        return np.min(eig_values)

    def ETA_BETAs(self, BETA, plot = False):
        """
        Computes QEBSG constants eta for several betas.

        Parameter:
            beta (numpy.ndarray): Multiple values of the smoothing parameter of SDG.
            plot (bool): Whether to plot the eta function of beta.

        Returns:
            numpy.ndarray: Array of QEBSG constants eta.
        """
        Eig = []
        for beta in BETA:
            Eig.append(self.eta_QEBSG(beta))
        Eig = np.array(Eig)
        if plot:
            plt.close()
            plt.semilogy(BETA, Eig)
            plt.xlabel(r'$\beta$', fontsize = 13)
            plt.ylabel(r'$\eta(\beta) = \lambda_{\min}(M_q)$', fontsize = 13)
            plt.grid(which = 'both')
            plt.show()
        return Eig

    def gamma_MSR(self):
        """
        Computes the MSR constant gamma.

        Returns:
            float: MSR constant gamma.
        """
        return np.min(abs(np.linalg.eig(self.matrix_M())[0]))

    def solve(self):
        """
        Solves the least squares problem using CVXPY.

        Returns:
            tuple: A tuple containing the optimal solution, optimal value, and problem status.
        """
        x = cp.Variable(self.n)
        objective = cp.Minimize(0.5*cp.norm(self.Q @ x - self.c)**2)
        constraints = [self.A @ x == self.b]
        prob = cp.Problem(objective, constraints)
        f_star = prob.solve()
        return x.value, f_star, prob.status

    def prob_parameters(self, verbose = False):
        """
        Computes some problem parameters.

        Parameter:
            verbose (bool): Whether to print verbose information.

        Returns:
            tuple: A tuple containing problem parameters such as:
                    * optimal solution.
                    * optimal value.
                    * Lipschitz constant of the gradient of the objective function.
                    * MSR constant gamma.
        """ 
        L = self.grad_Lipschitz()
        Lc_grad = self.gradc_Lipschitz()
        gamma = self.gamma_MSR()
        x_star, f_star, status = self.solve()
        if verbose: 
            print("Lipschitz constant of the gradient of the objective function = ", L)
            print("Lipschitz constant of the gradient of the conjugate function = ", Lc_grad)
            print("MSR-constant, gamma = ", gamma)
            print("Problem status: ", status)
            print('Optimal value using CVXPY = ', f_star)
        return x_star, f_star, L, gamma
    
class OneDim(LeastSquares):
    '''
    One-dimensional least-squares sub-class.

    Overridden methods:
        - solve: Solves the one-dimensional least squares problem "analytically".
        - prob_parameters: Computes problem parameters such as Lipschitz constants, MSR constant, optimal solution, etc...
                            A minor modification in a print statement if verbose.
    '''
    def solve(self):
        """
        Solves the one-dimensional least squares problem analytically.

        Returns:
            tuple: A tuple containing the optimal solution and optimal value.
        """
        x_star = self.b/self.A
        f_star = self.LS(x_star)
        return x_star, f_star

    def prob_parameters(self, verbose = False):
        """
        Computes some problem parameters.
            A minor modification in a print statement if verbose.

        Parameter:
            verbose (bool): Whether to print verbose information.

        Returns:
            tuple: A tuple containing problem parameters such as:
                    * optimal solution.
                    * optimal value.
                    * Lipschitz constant of the gradient of the objective function.
                    * MSR constant gamma.
        """ 
        L = self.grad_Lipschitz()
        Lc_grad = self.gradc_Lipschitz()
        gamma = self.gamma_MSR()
        x_star, f_star = self.solve()
        if verbose: 
            print("Lipschitz constant of the gradient of the objective function = ", L)
            print("Lipschitz constant of the gradient of the conjugate function = ", Lc_grad)
            print("MSR-constant, gamma = ", gamma)
            print('Analytical optimal value = ', f_star) 
            print('Analytical optimal solution = ', x_star)
        return x_star, f_star, L, gamma

class DistributedOPT(LeastSquares):
    '''
    Distributed optimization least-squares sub-class.

    New method:
        - data_processing: Splits the full dataset into sub-datasets. 
                           In our experiment, it means that we distribute the dataset across different computers.
        - constraint_matrix: Constructs the constraint matrix for the distributed optimization problem.
        - compact_Q: Constructs a compact Q matrix by arranging the matrices Q_1,...,Q_M along the diagonal.

    Overridden methods:
        - matrix_M: Constructs the MSR matrix for the distributed optimization problem.
        - solve: Solves the distributed optimization problem "analytically".
        - prob_parameters: Computes problem parameters such as Lipschitz constants, MSR constant, optimal solution, etc...
                            A minor modification in a print statement if verbose.
    '''
    def __init__(self, data, M, nbdata, nbfeatures):
        """
        Initializes an instance of the Distributed_OPT class.

        Attributes:
            M (int): Number of blocks.
            n (int): Dimension of the decision variable. 
            m (int): Number of features.
            Q (numpy.ndarray): Original Q matrix.
            c (numpy.ndarray): Vector c in the least squares objective.
            A (scipy.sparse.csr_matrix): Coefficient matrix of the linear constraints.
            b (numpy.ndarray): Right-hand side vector of the linear constraints.
        """
        self.data = data
        self.M = M
        self.nbdata = nbdata
        self.nbfeatures = nbfeatures
        self.n = nbfeatures   # Total number of variables.
        self.m = nbdata//M  # Total number of features
        self.Qorg, self.c = self.data_processing()  
        self.Q = self.compact_Q()  
        self.A = self.constraint_matrix()
        self.b = np.zeros((self.M-1)*self.n)
        super().__init__(self.n, self.m, self.Q, self.c, self.A, self.b)
  
    def data_processing(self):
        """
        Splits the full dataset into sub-datasets. 
            In our experiment, it means that we distribute the dataset across different computers.
        Returns:
            tuple: A tuple containing the processed matrix Q and vector c.
        """
        # 1. Read the text file and split it into rows
        with open(self.data, 'r') as file:
            rows = file.read().splitlines()
        
        # 2. Initialize lists to store the first column and the modified matrix
        c, Q = np.zeros(self.nbdata), np.zeros((self.nbdata, self.nbfeatures))

        # 3. Process each row and split it into values
        for i, row in enumerate(rows):
            values = row.split()
            
            # Append the first value (column c)
            c[i] = float(values[0])
            
            # Process and append the rest of the values (column Q)
            q_values = [v.split(':')[1] for v in values[1:]]
            Q[i] = q_values

        # 4. Return the processed data
        return Q.reshape(self.M, self.m, self.n), c
    
    def constraint_matrix(self):
        """
        Constructs the constraint matrix for the distributed optimization problem.

        Returns:
            scipy.sparse.csr_matrix: The constraint matrix.
        """
        # Create lists to store the row and column indices of the non-zero values
        row_indices = []
        col_indices = []

        # Generate random indices for each row where you want to place non-zero values
        for i in range((self.M-1)*self.n):
            row_indices.extend([i, i])
            col_indices.extend([i, i+self.n])

        # Create a list to store the values (which are all 1 in this case)
        data = np.tile(np.array([1, -1]), (self.M-1)*self.n)

        # Create the COO sparse matrix
        A = sp.coo_matrix((data, (row_indices, col_indices)), shape=((self.M-1)*self.n, self.M*self.n), dtype=float)
        return A    
    
    def compact_Q(self):
        """
        Constructs a compact Q matrix by arranging the matrices Q_1,...,Q_M along the diagonal.

        Returns:
            numpy.ndarray: The compact Q matrix.
        """
        Q = np.zeros((self.M*self.m, self.M*self.n), dtype=float) 
        for i in range(self.Qorg.shape[0]):
            # Insert the block Qi into the appropriate positions in the NumPy array
            row_indices = slice(i * self.m, (i + 1) * self.m)
            col_indices = slice(i * self.n, (i + 1) * self.n)
            Q[row_indices, col_indices] = self.Qorg[i]
        return Q
    
    def matrix_M(self):
        """
        Constructs the MSR matrix for the distributed optimization problem.

        Returns:
            numpy.ndarray: The MSR matrix.
        """
        s = (self.M - 1)*self.n
        zeros = np.zeros((s, s))
        return np.block([[self.Q.T@self.Q, self.A.toarray().T], [self.A.toarray(), zeros]])

    def solve(self):
        """
        Solves the distributed optimization problem "analytically".

        Returns:
            tuple: A tuple containing the optimal solution and optimal value.
        """
        P, v = 0, 0
        s = self.m
        for i in range(self.Qorg.shape[0]):
            P += self.Qorg[i].T @ self.Qorg[i]
            v += self.Qorg[i].T @ self.c[i*s : (i+1)*s]
        x_star = np.tile(np.linalg.solve(P, v), self.M)
        f_star = self.LS(x_star)
        return x_star, f_star

    def prob_parameters(self, verbose = False):
        """
        Computes some problem parameters.
            A minor modification in a print statement if verbose.

        Parameter:
            verbose (bool): Whether to print verbose information.

        Returns:
            tuple: A tuple containing problem parameters such as:
                    * optimal solution.
                    * optimal value.
                    * Lipschitz constant of the gradient of the objective function.
                    * MSR constant gamma.
        """ 
        L = self.grad_Lipschitz()
        Lc_grad = self.gradc_Lipschitz()
        gamma = self.gamma_MSR()
        x_star, f_star = self.solve()
        if verbose: 
            print("Lipschitz constant of the gradient of the objective function = ", L)
            print("Lipschitz constant of the gradient of the conjugate function = ", Lc_grad)
            print("MSR-constant, gamma = ", gamma)
            print('Analytical optimal value = ', f_star)
        return x_star, f_star, L, gamma
    
class QuadraticProgramming(LeastSquares):
    """
    Sub-class encapsulating all the necessary analysis of the Quadratic Programming problem, as presented is Subsection 7.3 in the article, for:
        - Solving the optimization problem: min ||Qx - c||^2 s.t. Ax = b and x >= 0.
            which is equivalent to: min ||Qx - c||^2 + i_{Rn_+}(x) s.t. Ax = b.  (EQP)
        - Computing the different measures we consider. 
        - Computing our findings bounds. 

    New Methods:
        - indicator: Computes the indicator function of Rn_+.
        - objective: The objective function of (EQP).
        - grad_LS: The gradient of the Least-Squares objective: ||Qx - c||^2.
        - prox_indicator: Computes the proximal operator of the indicator function of Rn_+.
        - prox_objective: Computes the proximal operator of the objective function of (EQP).
        - indicator_c: Computes the Fenchel-Conjugate of the indicator function of Rn_+.
        - objective_c: Computes the Fenchel-Conjugate of the objective function of (EQP).
        - proj_indicator_c: Computes the projection onto the domain of the Fenchel-Conjugate of the indicator function of Rn_+.
        - proj_objective_c: Computes the projection onto the domain of the Fenchel-Conjugate of the objective function of (EQP).
    
    Overridden Methods:
        - __init__: Initializes the Quadratic Programming problem with the given attributes.
        - stationarity: Computes the sub-differential of the Lagrangian w.r.t. x.
        - grad_Lipschitz: Returns +infinity since the objective function of (EQP) is non-smooth.
        - eta_QEBSG: Returns an estimate of the QEBSG constant eta. 
        - ETA_BETAs: Returns an estimate of the QEBSG constant eta for multiple values of beta.
        - gamma_MSR: Returns an estimate of the MSR constant gamma. 
        - solve: Solves the quadratic programming problem using CVXPY.
    """
    def __init__(self, n, m, Q, c, A, b, beta = None):
        """
        Initializes an instance of the QuadraticProgramming class with the given attributes.

        Attributes:
            n (int): Dimension of the decision variable.
            m (int): Number of the linear equality constraints.
            Q (numpy.ndarray): Coefficient matrix Q in the objective function.
            c (numpy.ndarray): Vector c in the least squares objective.
            A (numpy.ndarray): Coefficient matrix of the linear constraints.
            b (numpy.ndarray): Right-hand side vector of the constraints.
            beta (numpy.ndarray): Smoothing parameter of SDG.
        """
        self.n = n
        self.m = m
        self.Q = Q
        self.c = c
        self.A = A
        self.b = b
        self.beta = beta
        self.QQ_inv = np.linalg.pinv(self.Q.T @ self.Q)
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        check_sol = np.linalg.norm(A @ sol - b)
        print('Solution Existence:', check_sol < 1e-12, '\t', '|Ax - b| = ', check_sol)

    def indicator(self, x):
        """
        Computes the indicator function of Rn_+.

        Parameter:
            x (numpy.ndarray): Input vector.

        Returns:
            float: 0 if x >= 0 and +infinity otherwise.
        """
        if np.any(x < 0):
            return np.inf
        else:
            return 0
    
    def objective(self, x):
        """
        Computes the objective function of (EQP).

        Parameter:
            x (numpy.ndarray): Input vector.

        Returns:
            float: Value of the objective function.
        """
        return self.LS(x[:self.n]) + self.indicator(x[self.n:])

    def grad_LS(self, x):
        """
        Computes the gradient of the Least-Squares objective: ||Qx - c||^2.

        Parameter:
            x (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Gradient of the  Least-Squares objective.
        """
        return self.Q.T @ (self.Q @ x - self.c)
    
    def stationarity(self, x, y):
        """
        Computes the sub-differential of the Lagrangian w.r.t. x.

        Parameters:
            x (numpy.ndarray): Decision variable.
            y (numpy.ndarray): Lagrange multiplier (dual variable).

        Returns:
            numpy.ndarray: Gradient of the Lagrangian w.r.t. x.
        """
        q = x.copy()
        v = self.A.T @ y
        term1 = self.grad_LS(x[:self.n]) + v[:self.n]
        for i in range(self.n, 2*self.n):
            if x[i] == 0:
                q[i] = -1 * np.maximum(v[i], 0) + v[i]
            elif x[i] > 0:
                q[i] = v[i]
            else: 
                q[i] = np.inf
        return np.hstack((term1, q[self.n:]))
    
    def grad_Lipschitz(self):
        # Returns +infinity since the objective function of (EQP) is non-smooth.
        return np.inf
    
    def prox_indicator(self, x):
        """
        Computes the proximal operator of the indicator function of Rn_+.

        Parameter:
            x (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Proximal operator result.
        """
        return np.maximum(x, 0)
    
    def prox_objective(self, x, s):
        """
        Computes the proximal operator of the objective function of (EQP).

        Parameters:
            x (numpy.ndarray): Input vector.
            s (float): Step size.

        Returns:
            numpy.ndarray: Proximal operator result.
        """
        return np.hstack((self.prox_LS(x[:self.n], s), self.prox_indicator(x[self.n:])))
        
    def indicator_c(self, mu):
        """
        Computes the Fenchel-Conjugate of the indicator function of Rn_+.

        Parameter:
            mu (numpy.ndarray): Input vector.

        Returns:
            float: 0 if x <= 0 and +infinity otherwise.
        """
        return self.indicator(-1 * mu)
    
    def objective_c(self, mu):
        """
        Computes the Fenchel-Conjugate of the objective function of (EQP).

        Parameter:
            mu (numpy.ndarray): Input vector.

        Returns:
            float: Value of the Fenchel-Conjugate of the objective function of (EQP).
        """
        return self.LSc(mu[:self.n]) + self.indicator_c(mu[self.n:])
    
    def proj_indicator_c(self, a):
        """
        Computes the projection onto the domain of the Fenchel-Conjugate of the indicator function of Rn_+.

        Parameter:
            a (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Projection result.
        """
        return np.minimum(a, 0)
    
    def proj_objective_c(self, a):
        """
        Computes the projection onto the domain of the Fenchel-Conjugate of the objective function of (EQP).

        Parameter:
            a (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Projection result.
        """
        return np.hstack((self.proj_LSc(a[:self.n]), self.proj_indicator_c(a[self.n:])))
        
    def eta_QEBSG(self, beta):
        # Returns an estimate of the QEBSG constant eta. 
        return 1e-8
    
    def ETA_BETAs(self, BETAs):
        # Returns an estimate of the QEBSG constant eta for multiple values of beta.
        return 1e-8*np.ones(len(BETAs))

    def gamma_MSR(self):
        # Returns an estimate of the MSR constant gamma. 
        return 1e-8
    
    def solve(self):
        """
        Solves the quadratic programming problem using CVXPY.

        Returns:
            tuple: A tuple containing the optimal solution, optimal value, and problem status.
        """
        A_copy = self.A.copy()
        x = cp.Variable(self.n)
        objective = cp.Minimize(0.5 * cp.sum_squares(self.Q @ x - self.c))
        constraints = [0 <= x, A_copy[:self.m, :self.n] @ x == self.b[:self.m]]
        prob = cp.Problem(objective, constraints)
        f_star = prob.solve(solver = cp.OSQP, eps_abs = 1e-10, eps_rel = 1e-10)
        return x.value, f_star, prob.status

class BasisPursuit:
    """
    Class encapsulating all the necessary analysis of the Basis Pursuit problem, as presented is Subsection 7.4 in the article, for:
        - Solving the optimization problem: min ||x||_1 s.t. Ax = b. 
        - Computing the different measures we consider. 
        - Computing our findings bounds. 

    Methods:
        - __init__: Initializes the Basis Pursuit problem with the given attributes.
        - L1N: Computes the L1-norm of a vector.
        - stationarity: Computes the sub-differential of the Lagrangian w.r.t. x.
        - prox_L1N: Computes the proximal operator of the L1-norm function.
        - L1Nc: Computes the Fenchel-Conjugate of the L1-norm function.
        - proj_L1Nc: Computes the projection onto the domain of the Fenchel-Conjugate of the L1-norm function.
        - grad_Lipschitz: Returns +infinity since the objective function of (EQP) is non-smooth.
        - eta_QEBSG: Returns an estimate of the QEBSG constant eta. 
        - ETA_BETAs: Returns an estimate of the QEBSG constant eta for multiple values of beta.
        - gamma_MSR: Returns an estimate of the MSR constant gamma. 
        - solve: Solves the basis pursuit problem using CVXPY.
        - prob_parameters: Computes some problem parameters.
    """
    def __init__(self, n, m, A, b, beta = None):
        """
        Initializes an instance of the BasisPursuit class.

        Attributes:
            n (int): Dimension of the decision variable.
            m (int): Number of constraints.
            A (numpy.ndarray): Coefficient matrix of the linear constraints.
            b (numpy.ndarray): Right-hand side vector of the constraints.
            beta (numpy.ndarray): Smoothing parameter of SDG.
        """
        self.n = n
        self.m = m
        self.A = A
        self.b = b
        self.beta = beta
        check_sol = np.linalg.norm(A @ np.linalg.lstsq(A, b, rcond=None)[0] - b)
        print('Solution Existence:', check_sol < 1e-12, '\t', '|Ax - b| = ', check_sol)

    def L1N(self, x): 
        """
        Computes the L1-norm of a vector.

        Parameter:
            x (numpy.ndarray): Input vector.

        Returns:
            float: L1-norm of the vector.
        """
        return np.linalg.norm(x, 1)
    
    def stationarity(self, x, y):
        """
        Computes the sub-differential of the Lagrangian w.r.t. x.

        Parameters:
            x (numpy.ndarray): Decision variable.
            y (numpy.ndarray): Lagrange multiplier (dual variable).

        Returns:
            numpy.ndarray: Gradient of the Lagrangian w.r.t. x.
        """
        q = x.copy()
        v = self.A.T @ y
        for i in range(self.n):
            if x[i] == 0:
                q[i] = np.minimum(np.maximum(-v[i], -1), 1) + v[i]
            else: 
                q[i] = np.sign(x[i]) + v[i]
        return q
    
    def prox_L1N(self, x, s):
        """
        Computes the proximal operator of the L1-norm function.

        Parameters:
            x (numpy.ndarray): Input vector.
            s (float): Step size.

        Returns:
            numpy.ndarray: Proximal operator result.
        """
        res = np.maximum(np.abs(x) - s, 0)*np.sign(x)
        return res
    
    def L1Nc(self, mu):
        """
        Computes the Fenchel-Conjugate of the L1-norm function.

        Parameter:
            mu (numpy.ndarray): Input vector.

        Returns:
            float: Value of the Fenchel-Conjugate of the L1-norm function.
        """
        if np.linalg.norm(mu, np.inf) <= 1:
            return 0
        else: 
            return np.inf
        
    def proj_L1Nc(self, nu):
        """
        Computes the projection onto the domain of the Fenchel-Conjugate of the L1-norm function.

        Parameter:
            nu (numpy.ndarray): Input vector.

        Returns:
            numpy.ndarray: Projection result.
        """
        a = np.zeros_like(nu)
        for i in range(self.n):
            if np.abs(nu[i]) <= 1:
                a[i] = nu[i]
            else:
                a[i] = np.sign(nu[i])
        return a     
    
    def grad_Lipschitz(self):
        # The objective function ||x||_1 is non-smooth.
        return np.inf
    
    def eta_QEBSG(self, beta):
        # Returns an estimate of the QEBSG constant eta. 
        return 1e-8
    
    def ETA_BETAs(self, BETAs):
        # Returns an estimate of the QEBSG constant eta for multiple values of beta.
        return 1e-8*np.ones(len(BETAs))

    def gamma_MSR(self):
        # Returns an estimate of the MSR constant gamma. 
        return 1e-8
    
    def solve(self):
        """
        Solves the quadratic programming problem using CVXPY.

        Returns:
            tuple: A tuple containing the optimal solution, optimal value, and problem status.
        """
        x = cp.Variable(self.n)
        objective = cp.Minimize(cp.norm(x, 1))
        constraints = [self.A @ x == self.b]
        prob = cp.Problem(objective, constraints)
        f_star = prob.solve()
        return x.value, f_star, prob.status
    
    def prob_parameters(self, verbose = False):
        """
        Computes some problem parameters.

        Parameter:
            verbose (bool): Whether to print verbose information.

        Returns:
            tuple: A tuple containing problem parameters such as:
                    * optimal solution.
                    * optimal value.
                    * Lipschitz constant of the gradient of the objective function.
                    * MSR constant gamma.
        """ 
        L = self.grad_Lipschitz()
        gamma = self.gamma_MSR()
        x_star, f_star, status = self.solve()
        if verbose: 
            print("Lipschitz constant of the gradient of the objective function = ", L)
            print("MSR-constant, gamma = ", gamma)
            print("Problem status: ", status)
            print('Optimal value using CVXPY = ', f_star)
        return x_star, f_star, L, gamma