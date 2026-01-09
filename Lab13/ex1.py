"""
Lab 13 - Model Comparison
Exercise 1: Polynomial Model Analysis
Based on Lecture 11 material
"""

import pymc as pm
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import arviz as az

az.style.use('arviz-darkgrid')

# Load the data
dummy_data = np.loadtxt('./date.csv')
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]

# ============================================================================
# Exercise 1.1: Change order to 5 and perform inference
# ============================================================================

print("=" * 80)
print("Exercise 1.1: Polynomial model with order=5")
print("=" * 80)

order = 5

# Prepare polynomial features
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

# 1.1a: Perform inference with model_p (order=5, sd=10)
print("\n1.1a: Model with order=5, beta sd=10")
with pm.Model() as model_p_5:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order)
    ϵ = pm.HalfNormal('ϵ', 5)
    µ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
    idata_p_5 = pm.sample(2000, return_inferencedata=True, random_seed=42)

# Plot the curve for model with sd=10
x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)
x_new_p = np.vstack([x_new**i for i in range(1, order+1)])
x_new_s = (x_new_p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)

α_p_post = idata_p_5.posterior['α'].mean(("chain", "draw")).values
β_p_post = idata_p_5.posterior['β'].mean(("chain", "draw")).values
y_p_post = α_p_post + np.dot(β_p_post, x_new_s)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_1s[0], y_1s, c='C0', marker='.', label='Data')
plt.plot(x_new, y_p_post, 'C2', label=f'Order {order}, sd=10')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Model (order=5, beta sd=10)')
plt.legend()

# 1.1b: Repeat with sd=100
print("\n1.1b: Model with order=5, beta sd=100")
with pm.Model() as model_p_5_sd100:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=100, shape=order)
    ϵ = pm.HalfNormal('ϵ', 5)
    µ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
    idata_p_5_sd100 = pm.sample(2000, return_inferencedata=True, random_seed=42)

α_p_post_100 = idata_p_5_sd100.posterior['α'].mean(("chain", "draw")).values
β_p_post_100 = idata_p_5_sd100.posterior['β'].mean(("chain", "draw")).values
y_p_post_100 = α_p_post_100 + np.dot(β_p_post_100, x_new_s)

plt.subplot(1, 2, 2)
plt.scatter(x_1s[0], y_1s, c='C0', marker='.', label='Data')
plt.plot(x_new, y_p_post, 'C2', label=f'sd=10')
plt.plot(x_new, y_p_post_100, 'C3', label=f'sd=100')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison: sd=10 vs sd=100')
plt.legend()
plt.tight_layout()
plt.savefig('ex1_1_comparison_sd.png', dpi=150)
plt.show()

print("\n1.1b (continued): Model with variable sd=[10, 0.1, 0.1, 0.1, 0.1]")
with pm.Model() as model_p_5_var_sd:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=np.array([10, 0.1, 0.1, 0.1, 0.1]), shape=order)
    ϵ = pm.HalfNormal('ϵ', 5)
    µ = α + pm.math.dot(β, x_1s)
    y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
    idata_p_5_var = pm.sample(2000, return_inferencedata=True, random_seed=42)

α_p_post_var = idata_p_5_var.posterior['α'].mean(("chain", "draw")).values
β_p_post_var = idata_p_5_var.posterior['β'].mean(("chain", "draw")).values
y_p_post_var = α_p_post_var + np.dot(β_p_post_var, x_new_s)

plt.figure(figsize=(10, 6))
plt.scatter(x_1s[0], y_1s, c='C0', marker='.', label='Data', s=50)
plt.plot(x_new, y_p_post, 'C2', label='sd=10 (all)', linewidth=2)
plt.plot(x_new, y_p_post_100, 'C3', label='sd=100 (all)', linewidth=2)
plt.plot(x_new, y_p_post_var, 'C4', label='sd=[10, 0.1, 0.1, 0.1, 0.1]', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Different Prior Standard Deviations')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ex1_1b_all_comparisons.png', dpi=150)
plt.show()

print("\n" + "=" * 80)
print("Discussion:")
print("- sd=100: Wider prior allows more flexibility, potentially leading to overfitting")
print("- sd=[10, 0.1, ...]: Strong regularization on higher-order terms")
print("  This effectively creates a simpler model by constraining complex terms")
print("=" * 80)

# ============================================================================
# Exercise 1.2: Increase data points to 500
# ============================================================================

print("\n" + "=" * 80)
print("Exercise 1.2: Repeat with 500 data points")
print("=" * 80)

# Generate more data (simulating from the same pattern)
np.random.seed(42)
n_points = 500
x_1_large = np.random.uniform(x_1.min(), x_1.max(), n_points)
x_1_large = np.sort(x_1_large)

# Simulate y values with a polynomial relationship plus noise
# Using a quadratic relationship as suggested by the data
true_coeffs = [-0.5, -2.0]  # Approximate from the data pattern
y_1_large = (true_coeffs[0] * x_1_large +
             true_coeffs[1] * x_1_large**2 +
             np.random.normal(0, 1, n_points))

# Standardize
x_1p_large = np.vstack([x_1_large**i for i in range(1, order+1)])
x_1s_large = (x_1p_large - x_1p_large.mean(axis=1, keepdims=True)) / x_1p_large.std(axis=1, keepdims=True)
y_1s_large = (y_1_large - y_1_large.mean()) / y_1_large.std()

print("\nPerforming inference with 500 data points...")
with pm.Model() as model_p_5_large:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order)
    ϵ = pm.HalfNormal('ϵ', 5)
    µ = α + pm.math.dot(β, x_1s_large)
    y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s_large)
    idata_p_5_large = pm.sample(2000, return_inferencedata=True, random_seed=42)

# Plot results
x_new_large = np.linspace(x_1s_large[0].min(), x_1s_large[0].max(), 100)
x_new_p_large = np.vstack([x_new_large**i for i in range(1, order+1)])
x_new_s_large = (x_new_p_large - x_1p_large.mean(axis=1, keepdims=True)) / x_1p_large.std(axis=1, keepdims=True)

α_p_post_large = idata_p_5_large.posterior['α'].mean(("chain", "draw")).values
β_p_post_large = idata_p_5_large.posterior['β'].mean(("chain", "draw")).values
y_p_post_large = α_p_post_large + np.dot(β_p_post_large, x_new_s_large)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_1s[0], y_1s, c='C0', marker='.', label=f'Data (n={len(x_1)})', alpha=0.6)
plt.plot(x_new, y_p_post, 'C2', label='Model fit', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Original Data (n={len(x_1)})')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(x_1s_large[0], y_1s_large, c='C0', marker='.',
            label=f'Data (n={len(x_1_large)})', alpha=0.3, s=10)
plt.plot(x_new_large, y_p_post_large, 'C2', label='Model fit', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Large Dataset (n={len(x_1_large)})')
plt.legend()
plt.tight_layout()
plt.savefig('ex1_2_large_dataset.png', dpi=150)
plt.show()

print("\n" + "=" * 80)
print("Discussion:")
print("- With more data, the model becomes more stable")
print("- Posterior uncertainty decreases with larger sample size")
print("- The risk of overfitting is reduced with more observations")
print("=" * 80)

# ============================================================================
# Exercise 1.3: Cubic model (order=3) with WAIC and LOO comparison
# ============================================================================

print("\n" + "=" * 80)
print("Exercise 1.3: Cubic model comparison with WAIC and LOO")
print("=" * 80)

# Return to original data
x_1 = dummy_data[:, 0]
y_1 = dummy_data[:, 1]

# Linear model (order=1)
print("\nFitting linear model...")
order_l = 1
x_1p_l = np.vstack([x_1**i for i in range(1, order_l+1)])
x_1s_l = (x_1p_l - x_1p_l.mean(axis=1, keepdims=True)) / x_1p_l.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()

with pm.Model() as model_l:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10)
    ϵ = pm.HalfNormal('ϵ', 5)
    µ = α + β * x_1s_l[0]
    y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
    idata_l = pm.sample(2000, return_inferencedata=True, random_seed=42)

# Quadratic model (order=2)
print("\nFitting quadratic model...")
order_q = 2
x_1p_q = np.vstack([x_1**i for i in range(1, order_q+1)])
x_1s_q = (x_1p_q - x_1p_q.mean(axis=1, keepdims=True)) / x_1p_q.std(axis=1, keepdims=True)

with pm.Model() as model_q:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order_q)
    ϵ = pm.HalfNormal('ϵ', 5)
    µ = α + pm.math.dot(β, x_1s_q)
    y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
    idata_q = pm.sample(2000, return_inferencedata=True, random_seed=42)

# Cubic model (order=3)
print("\nFitting cubic model...")
order_c = 3
x_1p_c = np.vstack([x_1**i for i in range(1, order_c+1)])
x_1s_c = (x_1p_c - x_1p_c.mean(axis=1, keepdims=True)) / x_1p_c.std(axis=1, keepdims=True)

with pm.Model() as model_c:
    α = pm.Normal('α', mu=0, sigma=1)
    β = pm.Normal('β', mu=0, sigma=10, shape=order_c)
    ϵ = pm.HalfNormal('ϵ', 5)
    µ = α + pm.math.dot(β, x_1s_c)
    y_pred = pm.Normal('y_pred', mu=µ, sigma=ϵ, observed=y_1s)
    idata_c = pm.sample(2000, return_inferencedata=True, random_seed=42)

# Calculate WAIC and LOO
print("\nCalculating WAIC and LOO...")
pm.compute_log_likelihood(idata_l, model=model_l)
pm.compute_log_likelihood(idata_q, model=model_q)
pm.compute_log_likelihood(idata_c, model=model_c)

waic_l = az.waic(idata_l, scale="deviance")
waic_q = az.waic(idata_q, scale="deviance")
waic_c = az.waic(idata_c, scale="deviance")

loo_l = az.loo(idata_l, scale="deviance")
loo_q = az.loo(idata_q, scale="deviance")
loo_c = az.loo(idata_c, scale="deviance")

print("\n" + "=" * 80)
print("WAIC Results:")
print("=" * 80)
print(f"Linear model (order=1): {waic_l.elpd_waic:.2f} ± {waic_l.se:.2f}")
print(f"Quadratic model (order=2): {waic_q.elpd_waic:.2f} ± {waic_q.se:.2f}")
print(f"Cubic model (order=3): {waic_c.elpd_waic:.2f} ± {waic_c.se:.2f}")

print("\n" + "=" * 80)
print("LOO Results:")
print("=" * 80)
print(f"Linear model (order=1): {loo_l.elpd_loo:.2f} ± {loo_l.se:.2f}")
print(f"Quadratic model (order=2): {loo_q.elpd_loo:.2f} ± {loo_q.se:.2f}")
print(f"Cubic model (order=3): {loo_c.elpd_loo:.2f} ± {loo_c.se:.2f}")

# Model comparison with az.compare
cmp_df = az.compare({
    'Linear': idata_l,
    'Quadratic': idata_q,
    'Cubic': idata_c
}, ic="waic", scale="deviance")

print("\n" + "=" * 80)
print("Model Comparison (WAIC):")
print("=" * 80)
print(cmp_df)

# Plot comparison
az.plot_compare(cmp_df)
plt.title('Model Comparison: Linear vs Quadratic vs Cubic')
plt.tight_layout()
plt.savefig('ex1_3_model_comparison.png', dpi=150)
plt.show()

# Plot all three models
x_new = np.linspace(x_1.min(), x_1.max(), 100)

# Linear
x_new_l = (x_new - x_1.mean()) / x_1.std()
α_l_post = idata_l.posterior['α'].mean(("chain", "draw")).values
β_l_post = idata_l.posterior['β'].mean(("chain", "draw")).values
y_l_post = α_l_post + β_l_post * ((x_new - x_1.mean()) / x_1.std())

# Quadratic
x_new_p_q = np.vstack([x_new**i for i in range(1, order_q+1)])
x_new_s_q = (x_new_p_q - x_1p_q.mean(axis=1, keepdims=True)) / x_1p_q.std(axis=1, keepdims=True)
α_q_post = idata_q.posterior['α'].mean(("chain", "draw")).values
β_q_post = idata_q.posterior['β'].mean(("chain", "draw")).values
y_q_post = α_q_post + np.dot(β_q_post, x_new_s_q)

# Cubic
x_new_p_c = np.vstack([x_new**i for i in range(1, order_c+1)])
x_new_s_c = (x_new_p_c - x_1p_c.mean(axis=1, keepdims=True)) / x_1p_c.std(axis=1, keepdims=True)
α_c_post = idata_c.posterior['α'].mean(("chain", "draw")).values
β_c_post = idata_c.posterior['β'].mean(("chain", "draw")).values
y_c_post = α_c_post + np.dot(β_c_post, x_new_s_c)

plt.figure(figsize=(12, 6))
plt.scatter(x_1, y_1, c='C0', marker='.', label='Data', s=50, zorder=3)
plt.plot(x_new, y_l_post * y_1.std() + y_1.mean(), 'C1',
         label='Linear (order=1)', linewidth=2)
plt.plot(x_new, y_q_post * y_1.std() + y_1.mean(), 'C2',
         label='Quadratic (order=2)', linewidth=2)
plt.plot(x_new, y_c_post * y_1.std() + y_1.mean(), 'C3',
         label='Cubic (order=3)', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Comparison of Linear, Quadratic, and Cubic Models')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ex1_3_all_models_plot.png', dpi=150)
plt.show()

print("\n" + "=" * 80)
print("Final Discussion:")
print("=" * 80)
print("Based on WAIC and LOO:")
print("- Lower values indicate better predictive performance")
print("- The best model balances fit and complexity")
print("- Check the 'weight' column to see model probabilities")
print("- The 'dse' (standard error of difference) helps assess significance")
print("=" * 80)
print("\nAll plots have been saved!")
print("- ex1_1_comparison_sd.png")
print("- ex1_1b_all_comparisons.png")
print("- ex1_2_large_dataset.png")
print("- ex1_3_model_comparison.png")
print("- ex1_3_all_models_plot.png")
print("=" * 80)
