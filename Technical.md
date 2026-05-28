# [SYSTEM_PROMPT]
# ROLE: Senior Data/ML Engineer Agent
# OBJECTIVE: Implement mathematical models for Data Fitting & OLS strictly from scratch using NumPy.
# CONSTRAINT_1: DO NOT use `sklearn.linear_model`, `scipy.optimize`, or `numpy.linalg.lstsq` for core regression tasks.
# CONSTRAINT_2: Ensure exact reproducibility (`np.random.seed(42)`).
# CONSTRAINT_3: Prevent Data Leakage (Strict isolation of fit/transform states).

---

## 1. [ENVIRONMENT_SETUP]
**OS Target:** macOS (Apple Silicon M1) / Unix.
**Environment Manager:** `micromamba` (Avoid `venv` to prevent PATH separator issues).

```bash
micromamba create -n toan_udtk python=3.10
micromamba activate toan_udtk
pip install numpy pandas matplotlib seaborn scikit-learn
## 2. [MODULE_1]: CORE_MATH_IMPLEMENTATION
**Path:** `part1/ols_implementation.py`
**Dependencies:** `numpy`

### Task 1.1: Ordinary Least Squares (OLS)
* **Input:** Design matrix $X \in \mathbb{R}^{n \times p}$, Target vector $y \in \mathbb{R}^n$.
* **Algorithm:** 1. Compute coefficients: $\hat{\beta} = (X^T X)^{-1} X^T y$
    2. Compute Hat Matrix: $H = X(X^T X)^{-1} X^T$
    3. Verify Idempotence: Assert $H \cdot H \approx H$ (within floating-point tolerance).
* **Output:** `beta_hat` (array), `H` (matrix).

### Task 1.2: Statistical Inference & Metrics
* **Input:** $X, y, \hat{\beta}$
* **Algorithm:**
    1. Residuals: $\hat{e} = y - X\hat{\beta}$
    2. Variance of residuals: $\hat{\sigma}^2 = \frac{\hat{e}^T \hat{e}}{n - p}$
    3. Variance of $\hat{\beta}$: $\text{Var}(\hat{\beta}) = \hat{\sigma}^2 (X^T X)^{-1}$
    4. t-statistics: $t_j = \frac{\hat{\beta}_j}{\sqrt{\text{Var}(\hat{\beta})_{jj}}}$
    5. Goodness-of-fit: Compute $R^2$ and Adjusted $R^2$.
* **Output:** Dictionary containing `R2`, `Adj_R2`, `t_stats`, `residuals`.

---

## 3. [MODULE_2]: REGULARIZATION_ALGORITHMS
**Path:** `part1/ridge_lasso.py`

### Task 2.1: Ridge Regression (L2 Penalty)
* **Algorithm:** $\hat{\beta}_{ridge} = (X^T X + \lambda I_p)^{-1} X^T y$
* **Constraint:** Do not penalize the intercept $\beta_0$. The first diagonal element of $I_p$ must be $0$.

### Task 2.2: Lasso Regression (L1 Penalty) - Coordinate Descent
* **Algorithm:** No closed-form. Implement iterative Coordinate Descent.
    1. Initialize $\beta = \beta_{OLS}$ or zeros.
    2. For each feature $j \in \{1 \dots p\}$:
       Compute partial residual $\rho_j = X_j^T (y - X_{-j}\beta_{-j})$
       Compute normalizer $z_j = X_j^T X_j$
    3. Soft-thresholding update: 
       $\beta_j = \frac{1}{z_j} \text{sign}(\rho_j) \max(0, |\rho_j| - \lambda)$
    4. Repeat until convergence ($||\beta^{(t)} - \beta^{(t-1)}||_2 < \epsilon$).

---

## 4. [MODULE_3]: ADVANCED_TECHNIQUES (BONUS)
**Path:** `part1/advanced_methods.py`
**Objective:** Implement non-standard regression techniques to demonstrate algebraic mastery.

### Task 3.1: Elastic Net (L1 + L2 Hybrid)
* **Algorithm:** Modify Coordinate Descent for Elastic Net.
    Update rule: $\beta_j = \frac{\text{sign}(\rho_j) \max(0, |\rho_j| - \lambda_1)}{z_j + \lambda_2}$
* **Output:** Method `elastic_net_fit(X, y, l1_ratio, alpha)`.

### Task 3.2: Robust Regression (Huber Loss via IRLS)
* **Objective:** Mitigate the effect of outliers in heavy-tailed distributions.
* **Algorithm (Iteratively Reweighted Least Squares):**
    1. Initialize weights $W = I_n$ (Identity matrix).
    2. Repeat until convergence:
       a. $\hat{\beta} = (X^T W X)^{-1} X^T W y$
       b. Compute residuals $e = y - X\hat{\beta}$
       c. Update diagonal weights $W_{ii}$:
          If $|e_i| \le \delta$: $W_{ii} = 1$
          If $|e_i| > \delta$: $W_{ii} = \frac{\delta}{|e_i|}$

### Task 3.3: Bayesian Linear Regression
* **Prior:** $P(\beta) = \mathcal{N}(0, \alpha^{-1}I)$
* **Likelihood:** $P(y|X,\beta) = \mathcal{N}(X\beta, \beta_n^{-1}I)$
* **Algorithm:** Compute exact Posterior parameters.
    Covariance: $\Sigma_N = (\alpha I + \beta_n X^T X)^{-1}$
    Mean: $\mu_N = \beta_n \Sigma_N X^T y$
* **Output:** Posterior mean vector and covariance matrix for credible intervals.

---

## 5. [MODULE_4]: OOP_DATA_PIPELINE
**Path:** `part2/data_pipeline.py`
**Objective:** Object-oriented robust pre-processing.

* **Class:** `NBADataPipeline`
* **State Variables:** `self.scaler_params`, `self.dummy_cols`, `self.imputer` (sklearn `KNNImputer(n_neighbors=5, weights='distance')`).
* **Method `fit(X)`:**
    1. Isolate numeric columns. Compute mean and standard deviation for Z-score normalization.
    2. Fit K-NN Imputer on scaled data.
    3. Compute Dummy Variables layout using `pd.get_dummies(..., drop_first=True)`. Store column order.
* **Method `transform(X)`:**
    1. Target transform: If `Salary` exists, $y' = \ln(\text{Salary})$.
    2. Standardize numeric columns using `fit` state.
    3. Impute `NaN` values using K-NN.
    4. Encode categorical variables. 
    5. **CRITICAL:** Use `df.reindex(columns=self.dummy_cols, fill_value=0)` to align test set dimensions strictly to train set dimensions.
    6. Return $X_{processed}, y_{processed}$.

---

## 6. [EXECUTION_FLOW]
1. Execute `ols_implementation.py` via unit tests to assert matrix integrity.
2. Build `NBADataPipeline` and run $80/20$ train-test split.
3. Pass `X_train_processed, y_train` to OLS, Ridge, Lasso, and Advanced Methods.
4. Evaluate on `X_test_processed`.
5. Reverse log-transform for evaluation: $\text{RMSE} = \sqrt{\frac{1}{n} \sum (\exp(y) - \exp(\hat{y}))^2}$.