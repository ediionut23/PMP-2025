"""
Exercitiul 1: Analiza relatiei dintre orele de exercitii fizice pe saptamana
si nivelurile de colesterol folosind modele de mixtura cu regresie polinomiala.

Modelul presupune ca fiecare individ apartine uneia dintre K subpopulatii,
fiecare avand propriul model de regresie polinomiala:
    Cholesterol_i = sum_{k=1}^{K} w_k * N(mu_{k,i}, sigma_k^2)
    unde mu_{k,i} = alpha_k + beta_k * t_i + gamma_k * t_i^2
"""

import os
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import pytensor.tensor as pt
from scipy import stats

np.random.seed(42)

script_dir = os.path.dirname(os.path.abspath(__file__))
data = pd.read_csv(os.path.join(script_dir, 'date_colesterol.csv'))
t = data['Ore_Exercitii'].values
y = data['Colesterol'].values
n_obs = len(y)

t_mean = t.mean()
t_std = t.std()
t_norm = (t - t_mean) / t_std


def build_mixture_regression_model(K, t_data, y_data):
    """
    Construieste un model de mixtura cu regresie polinomiala de gradul 2.

    Parametri:
        K: int - numarul de subpopulatii (componente ale mixturii)
        t_data: array - predictorul normalizat (ore exercitii)
        y_data: array - variabila observata (colesterol)

    Returneaza:
        model: PyMC Model - modelul Bayesian
    """
    with pm.Model() as model:
        w = pm.Dirichlet('w', a=np.ones(K))

        init_alphas = np.linspace(y_data.min(), y_data.max(), K)

        alpha = pm.Normal('alpha', mu=init_alphas, sigma=20, shape=K,
                         transform=pm.distributions.transforms.ordered)

        beta = pm.Normal('beta', mu=0, sigma=10, shape=K)

        gamma = pm.Normal('gamma', mu=0, sigma=5, shape=K)

        sigma = pm.HalfNormal('sigma', sigma=15)

        def logp_mixture(y_obs):
            logps = []
            for k in range(K):
                mu_k = alpha[k] + beta[k] * t_data + gamma[k] * t_data**2
                logp_k = pm.logp(pm.Normal.dist(mu=mu_k, sigma=sigma), y_obs)
                logps.append(logp_k + pt.log(w[k]))

            return pm.math.logsumexp(pt.stack(logps, axis=0), axis=0).sum()

        pm.Potential('likelihood', logp_mixture(y_data))

    return model


"""
Subpunctul 1: Estimare parametri pentru K=3,4,5.

Se estimeaza folosind inferenta Bayesiana (MCMC) ponderile w_k si
coeficientii de regresie (alpha_k, beta_k, gamma_k) pentru fiecare
subpopulatie, pentru fiecare valoare K in {3, 4, 5}.
"""

clusters = [3, 4, 5]
models = {}
idatas = {}

print("=" * 70)
print("SUBPUNCTUL 1: Estimarea parametrilor pentru K = 3, 4, 5")
print("=" * 70)

for K in clusters:
    print(f"\n{'='*50}")
    print(f"Fitting model cu K = {K} componente...")
    print(f"{'='*50}")

    model = build_mixture_regression_model(K, t_norm, y)

    with model:
        idata = pm.sample(2000, tune=2000, target_accept=0.95,
                         random_seed=42, return_inferencedata=True,
                         cores=1)

    models[K] = model
    idatas[K] = idata

    print(f"\n--- Rezultate pentru K = {K} ---")
    print("\nPonderi (w):")
    w_summary = az.summary(idata, var_names=['w'])
    print(w_summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])

    print("\nCoeficienti alpha (intercept):")
    alpha_summary = az.summary(idata, var_names=['alpha'])
    print(alpha_summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])

    print("\nCoeficienti beta (liniar):")
    beta_summary = az.summary(idata, var_names=['beta'])
    print(beta_summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])

    print("\nCoeficienti gamma (patratic):")
    gamma_summary = az.summary(idata, var_names=['gamma'])
    print(gamma_summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])

    print("\nDeviatie standard (sigma):")
    sigma_summary = az.summary(idata, var_names=['sigma'])
    print(sigma_summary[['mean', 'sd', 'hdi_3%', 'hdi_97%']])


"""
Subpunctul 2: Selectia numarului optim K de subpopulatii.

Se compara modelele cu K=3,4,5 folosind criteriile Bayesiene WAIC
(Widely Applicable Information Criterion) si LOO (Leave-One-Out
Cross-Validation). Modelul cu valoarea cea mai mica (pentru WAIC/LOO)
este preferat.

Nota: Deoarece folosim pm.Potential pentru likelihood, vom calcula
WAIC manual folosind log-likelihood din esantioanele posterior.
"""

print("\n" + "=" * 70)
print("SUBPUNCTUL 2: Compararea modelelor folosind WAIC")
print("=" * 70)


def compute_log_lik_manual(idata, K, t_data, y_data):
    """
    Calculeaza log-likelihood pentru fiecare observatie si esantion.

    Parametri:
        idata: InferenceData - rezultatele MCMC
        K: int - numarul de componente
        t_data: array - predictorul normalizat
        y_data: array - observatiile

    Returneaza:
        log_lik: array de forma (n_samples, n_obs)
    """
    posterior = idata.posterior.stack(samples=("chain", "draw"))
    n_samples = posterior.samples.size
    n_obs = len(y_data)

    log_lik = np.zeros((n_samples, n_obs))

    for s in range(n_samples):
        w_s = posterior['w'][:, s].values
        alpha_s = posterior['alpha'][:, s].values
        beta_s = posterior['beta'][:, s].values
        gamma_s = posterior['gamma'][:, s].values
        sigma_s = float(posterior['sigma'][s].values)

        for i in range(n_obs):
            p_mix = 0
            for k in range(K):
                mu_k = alpha_s[k] + beta_s[k] * t_data[i] + gamma_s[k] * t_data[i]**2
                p_mix += w_s[k] * stats.norm.pdf(y_data[i], mu_k, sigma_s)
            log_lik[s, i] = np.log(p_mix + 1e-10)

    return log_lik


def compute_waic(log_lik):
    """
    Calculeaza WAIC din log-likelihood.

    WAIC = -2 * (lppd - p_waic)
    unde:
        lppd = sum_i log(mean_s(p(y_i|theta_s)))
        p_waic = sum_i var_s(log p(y_i|theta_s))

    Parametri:
        log_lik: array (n_samples, n_obs) - log-likelihood per obs si sample

    Returneaza:
        waic: float - valoarea WAIC
        p_waic: float - numarul efectiv de parametri
    """
    lppd = np.sum(np.log(np.mean(np.exp(log_lik), axis=0)))

    p_waic = np.sum(np.var(log_lik, axis=0))

    waic = -2 * (lppd - p_waic)

    return waic, p_waic, lppd


print("\nCalcul WAIC pentru fiecare model...")
waic_results = {}

for K in clusters:
    print(f"  Calculez log-likelihood pentru K={K}...")
    log_lik = compute_log_lik_manual(idatas[K], K, t_norm, y)
    waic, p_waic, lppd = compute_waic(log_lik)
    waic_results[K] = {'waic': waic, 'p_waic': p_waic, 'lppd': lppd}
    print(f"    WAIC = {waic:.2f}, p_waic = {p_waic:.2f}")

print("\n" + "-" * 50)
print("REZULTATE WAIC")
print("-" * 50)
print(f"{'K':<5} {'WAIC':<12} {'p_WAIC':<12} {'lppd':<12}")
print("-" * 50)

for K in clusters:
    print(f"{K:<5} {waic_results[K]['waic']:<12.2f} {waic_results[K]['p_waic']:<12.2f} {waic_results[K]['lppd']:<12.2f}")

best_K = min(waic_results.keys(), key=lambda k: waic_results[k]['waic'])

print("\n" + "=" * 70)
print("CONCLUZII")
print("=" * 70)
print(f"\nConform WAIC, cel mai bun model este: K = {best_K}")

print("\n--- Justificare ---")
print(f"""
Criteriul WAIC (Widely Applicable Information Criterion) este o masura
a capacitatii predictive a modelului care penalizeaza complexitatea.

Rezultate:""")
for K in clusters:
    diff = waic_results[K]['waic'] - waic_results[best_K]['waic']
    print(f"  K={K}: WAIC = {waic_results[K]['waic']:.2f} (diferenta fata de best: {diff:.2f})")

print(f"""
Modelul cu K={best_K} are cea mai mica valoare WAIC, indicand cel mai
bun compromis intre fit-ul datelor si complexitatea modelului.

Interpretare:
- Valori mai mici ale WAIC indica modele cu capacitate predictiva mai buna
- p_WAIC estimeaza numarul efectiv de parametri (complexitatea modelului)
- Modelul selectat (K={best_K}) reprezinta numarul optim de subpopulatii
  pentru a descrie relatia dintre exercitii fizice si colesterol
""")

fig, ax = plt.subplots(figsize=(10, 6))

waics = [waic_results[K]['waic'] for K in clusters]
colors = ['green' if K == best_K else 'blue' for K in clusters]

bars = ax.bar([f'K={K}' for K in clusters], waics, color=colors, edgecolor='black')
ax.set_ylabel('WAIC (mai mic = mai bun)')
ax.set_xlabel('Numar de componente (K)')
ax.set_title('Comparare modele folosind WAIC')

for bar, waic in zip(bars, waics):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{waic:.1f}', ha='center', va='bottom')

ax.legend([bars[clusters.index(best_K)]], [f'Model optim (K={best_K})'])

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'comparare_waic.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n--- Vizualizare fit pentru modelul optim K={best_K} ---")

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(t, y, alpha=0.5, label='Date observate', s=20, c='gray')

posterior = idatas[best_K].posterior.stack(samples=("chain", "draw"))
t_plot = np.linspace(t.min(), t.max(), 100)
t_plot_norm = (t_plot - t_mean) / t_std

w_mean = posterior['w'].mean('samples').values
alpha_mean = posterior['alpha'].mean('samples').values
beta_mean = posterior['beta'].mean('samples').values
gamma_mean = posterior['gamma'].mean('samples').values
sigma_mean = float(posterior['sigma'].mean('samples').values)

colors = plt.cm.tab10(np.linspace(0, 1, best_K))
for k in range(best_K):
    y_k = alpha_mean[k] + beta_mean[k] * t_plot_norm + gamma_mean[k] * t_plot_norm**2
    ax.plot(t_plot, y_k, '--', color=colors[k], linewidth=2,
            label=f'Componenta {k+1} (w={w_mean[k]:.2f})')

y_mix = np.zeros_like(t_plot)
for k in range(best_K):
    y_k = alpha_mean[k] + beta_mean[k] * t_plot_norm + gamma_mean[k] * t_plot_norm**2
    y_mix += w_mean[k] * y_k

ax.plot(t_plot, y_mix, 'r-', linewidth=3, label='Media mixturii')

ax.set_xlabel('Ore exercitii pe saptamana')
ax.set_ylabel('Colesterol')
ax.set_title(f'Model de mixtura cu K={best_K} componente (regresie polinomiala)')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'fit_model_optim.png'), dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("REZUMAT FINAL")
print("=" * 70)
print(f"""
Problema analizata: Relatia dintre orele de exercitii fizice si colesterol,
modelata ca o mixtura de K subpopulatii cu regresii polinomiale de gradul 2.

Model: Cholesterol_i ~ sum_k w_k * N(alpha_k + beta_k*t + gamma_k*t^2, sigma^2)


Parametrii estimati pentru K={best_K}:
  - Ponderi (w): {np.round(w_mean, 3)}
  - Intercepti (alpha): {np.round(alpha_mean, 2)}
  - Coef. liniari (beta): {np.round(beta_mean, 2)}
  - Coef. patratici (gamma): {np.round(gamma_mean, 2)}
  - Deviatie standard: {sigma_mean:.2f}

""")
