import numpy as np
import pymc as pm
import arviz as az

# Datele din tabel 
publicity = np.array([1.5, 2.0, 2.3, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0,
                      6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0])

sales = np.array([5.2, 6.8, 7.5, 8.0, 9.0, 10.2, 11.5, 12.0, 13.5, 14.0,
                  15.0, 15.5, 16.2, 17.0, 18.0, 18.5, 19.5, 20.0, 21.0, 22.0])

new_publicity = np.array([3.0, 7.0, 12.0])

# Modelul Bayesian
with pm.Model() as model:
    
    alpha = pm.Normal("alpha", mu=0, sigma=20)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.HalfNormal("sigma", sigma=5)

    mu = alpha + beta * publicity

    pm.Normal("sales", mu=mu, sigma=sigma, observed=sales)

    idata = pm.sample(2000, tune=1000, target_accept=0.9, random_seed=42)

# HDI pentru coeficienți
coef_hdi = az.hdi(idata, var_names=["alpha", "beta"], hdi_prob=0.94)
print("\nCoefficient HDIs (94%):\n", coef_hdi)

with model:
    pm.set_data = None 
    post = idata.posterior

    # calculăm predicții manual
    alpha_s = post["alpha"].values.reshape(-1)
    beta_s  = post["beta"].values.reshape(-1)
    sigma_s = post["sigma"].values.reshape(-1)

preds = []
for x in new_publicity:
    mu_pred = alpha_s + beta_s * x
    y_samples = np.random.normal(mu_pred, sigma_s)
    preds.append(y_samples)

preds = np.array(preds)

# intervale 5%-95%
low = np.percentile(preds, 5, axis=1)
high = np.percentile(preds, 95, axis=1)

print("\nPredictive intervals (5th–95th):")
for p, lo, hi in zip(new_publicity, low, high):
    print(f"publicity={p:>4.1f} -> sales ≈ {lo:5.2f}  to  {hi:5.2f}")
