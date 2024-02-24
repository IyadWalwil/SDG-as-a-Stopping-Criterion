## Smoothed Duality Gap: A Stopping Criterion Implementation

This repository contains the code developed to validate and illustrate the theoretical findings outlined in the paper titled **"The Smoothed Duality Gap as a Stopping Criterion"** by Iyad Walwil and Olivier Fercoq.

### Smoothed Duality Gap
The *Smoothed Duality Gap*, initially introduced in `[3]`, represents a novel measure of optimality that is widely applicable but remains less studied compared to the previously discussed ones.

Given $\beta = (\beta_x, \beta_y) \in [0, +\infty]^2, z \in \mathcal{Z}$ and $\dot{z} \in \mathcal{Z}$, the smoothed gap $\mathcal{G}_{\beta}$ is the function defined by:

$$
\mathcal{G}_{\beta}(z;\dot{z}) = \\sup ~ \mathcal{L}(x, y') - \mathcal{L}(x', y) - \frac{\beta_x}{2} \\|x' - \dot{x}\\|^2 - \frac{\beta_y}{2} \\|y' - \dot{y}\\|^2
$$

Where the sup is taken over $z' = (x', y')$. When the smoothness parameter $\beta = 0$, we recover the conventional duality gap. The smoothed duality gap concept involves smoothing the duality gap through a proximity function `[2]`, thereby ensuring that the smoothed duality gap attains finite values for constrained problems, unlike its conventional counterpart. Additionally, when the smoothness parameter is small and the smoothed duality gap is small, it signifies that both the optimality gap and the feasibility error are also small.

Moreover, the author in `[1]` has found that the smoothed duality gap offers a robust outcome. Independently of any unknown or uncomputable variables, it serves as a valid optimality measure. Therefore,  it could be utilized as a stopping criterion.

### Modules Overview:

1. **`PDHG.py`**: This module implements the Primal-Dual Hybrid Gradient (PDHG) algorithm for solving optimization problems of the form:
   
    $$\min_{x \in \mathcal{X}} \max_{y \in \mathcal{Y}}~ f(x) + \left\langle Ax - b, y \right\rangle $$

   where:
   - $f \colon \mathcal{X} \rightarrow \mathbb{R} \cup \\{+\infty\\}$ is a proper, lower semi-continuous, convex function with a computable proximal operator.
   - $A \colon \mathcal{X} \rightarrow \mathcal{Y}$ is a linear operator.
   - $b \in \mathcal{Y}$.

   It provides users with four different stopping criteria options:
   - Fixed Number of Iterations **(FNoI)**.
   - Distance to the Optimum **(DttO)**.
   - Karush-Kuhn-Tucker **(KKT)** error.
   - Smoothed Duality Gap **(SDG)** - *Default*.

2. **`OptimizationMeasures.py`**: Computes the various optimality measures studied in Sections **3 & 4** of the paper:
   - Optimality Gap **(OG)**.
   - Feasibility Gap **(FG)**.
   - Karush-Kuhn-Tucker **(KKT)** error.
   - Projected Duality Gap **(PDG)**.
   - Smoothed Duality Gap **(SDG)**.

3. **`OptimizationBounds.py`**: Implements the theoretical findings presented in Sections **5 & 6**.

4. **`Plotter.py`**: Utilizes `matplotlib` to plot curves by taking a dictionary of curves along with their labels and titles.

5. **`OptimizationProblems.py`**: Encapsulates the analysis of the various optimization problems presented in Section **7**.

### Usage Instructions:

To experiment with the code, simply run the **`Experiments.ipynb`** file, which contains all the conducted experiments. This notebook initializes several instances of the problems defined in `OptimizationProblems.py`, solves them using `PDHG.py`, computes measures and bounds with `OptimizationBounds.py`, and finally visualizes the results using `Plotter.py`.

### Credits:

- **Produced by:** Iyad Walwil.
- **Supervised by:** Olivier Fercoq.


### References

`[1]`: Fercoq, O.: Quadratic error bound of the smoothed gap and the restarted averaged primal-dual hybrid gradient. Open Journal of Mathematical Optimization 4, 6 (2023). DOI 10.5802/ojmo.26. URL https://ojmo.centre-mersenne.org/articles/10.5802/ojmo.26/.

`[2]`: Nesterov, Y.: Smooth minimization of non-smooth functions Mathematical Programming 103, 127–152 (2005). URL https://api.semanticscholar.org/CorpusID:2391217.

`[3]`: Tran-Dinh, Q., Fercoq, O., Cevher, V.: A smooth primal-dual optimization framework for nonsmooth composite convex minimization. SIAM Journal on Optimization 28(1), 96–134 (2018). DOI 10.1137/16M1093094. URL https://doi.org/10.1137/16M1093094. 
