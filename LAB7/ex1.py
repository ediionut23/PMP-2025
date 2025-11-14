import numpy as np
import pymc as pm
import arviz as az

data = np.array([56, 60, 58, 55, 57, 59, 61, 56, 58, 60])
x = data.mean()

def hdi(samples, hdi_prob=0.95):
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_idx_inc = int(np.floor(hdi_prob * n))
    n_intervals = n - interval_idx_inc
    interval_width = sorted_samples[interval_idx_inc:] - sorted_samples[:n_intervals]
    min_idx = np.argmin(interval_width)
    return sorted_samples[min_idx], sorted_samples[min_idx + interval_idx_inc]

# === a) Model PyMC cu priors rezonabile ===
print(f"\na) x = {x:.2f} (media observata)")
print(f"   Alegem x = media observata pentru ca oferim un prior centrat pe date,")
print(f"   dar cu varianta suficient de mare (10^2) pentru a nu introduce bias.")
print(f"   Prior: mu ~ N({x:.2f}, 10^2), sigma ~ HalfNormal(10)")

with pm.Model() as model_weak:
    mu = pm.Normal('mu', mu=x, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=10)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
    trace_weak = pm.sample(2000, random_seed=123, return_inferencedata=True)

print("\n   Summary posterior:")
print(az.summary(trace_weak, hdi_prob=0.95))

mu_samples = trace_weak.posterior['mu'].values.flatten()
sigma_samples = trace_weak.posterior['sigma'].values.flatten()

# === b) 95% HDI ===
mu_hdi = hdi(mu_samples)
sigma_hdi = hdi(sigma_samples)

print(f"\nb) 95% HDI:")
print(f"   mu:    [{mu_hdi[0]:.2f}, {mu_hdi[1]:.2f}] dB")
print(f"   sigma: [{sigma_hdi[0]:.2f}, {sigma_hdi[1]:.2f}] dB")

mu_bayes = mu_samples.mean()
sigma_bayes = sigma_samples.mean()

# === c) Comparatie Bayesian vs Frequentist ===
mu_freq = data.mean()
sigma_freq = data.std(ddof=1)

print(f"\nc) Comparatie Bayesian vs Frequentist:")
print(f"   Bayesian:     mu = {mu_bayes:.2f} dB,  sigma = {sigma_bayes:.2f} dB")
print(f"   Frequentist:  mu = {mu_freq:.2f} dB,  sigma = {sigma_freq:.2f} dB")
print(f"   Diferente:    Δmu = {abs(mu_bayes - mu_freq):.4f} dB,  Δsigma = {abs(sigma_bayes - sigma_freq):.4f} dB")

print(f"\n   Diferentele sunt foarte mici, practic neglijabile. Acest lucru se intampla")
print(f"   pentru ca priorii alesi sunt suficient de vagi (variantele mari permit")
print(f"   datelor sa 'vorbeasca'). Cu 10 observatii, datele au suficienta greutate")
print(f"   pentru a domina inferenta, iar priorul nu introduce bias semnificativ.")

# === d) Model cu prior puternic ===
with pm.Model() as model_strong:
    mu = pm.Normal('mu', mu=50, sigma=1)
    sigma = pm.HalfNormal('sigma', sigma=10)
    y = pm.Normal('y', mu=mu, sigma=sigma, observed=data)
    trace_strong = pm.sample(2000, random_seed=123, return_inferencedata=True)

print(f"\nd) Model cu prior puternic N(50, 1^2):")
print("\n   Summary posterior:")
print(az.summary(trace_strong, hdi_prob=0.95))

mu_samples_strong = trace_strong.posterior['mu'].values.flatten()
sigma_samples_strong = trace_strong.posterior['sigma'].values.flatten()
mu_strong = mu_samples_strong.mean()
sigma_strong = sigma_samples_strong.mean()

print(f"\n   Comparatie:")
print(f"   Prior vag:      mu = {mu_bayes:.2f} dB")
print(f"   Prior puternic: mu = {mu_strong:.2f} dB")
print(f"   Frequentist:    mu = {mu_freq:.2f} dB")
print(f"   Diferenta:      {abs(mu_strong - mu_bayes):.2f} dB")

print(f"\n   Explicatie:")
print(f"   Priorul puternic N(50, 1^2) are varianta foarte mica (sigma=1), ceea ce")
print(f"   inseamna ca 'crede' puternic ca mu este aproape de 50. Insa datele noastre")
print(f"   au media {mu_freq:.2f}, deci exista un conflict intre prior si date.")
print(f"   Posteriorul rezulta din compromisul dintre cele doua: este 'tras' spre 50")
print(f"   de prior, dar si influentat de date. Rezultatul ({mu_strong:.2f}) este mai")
print(f"   aproape de prior decat de date, pentru ca priorul este foarte informativ.")
print(f"   De asemenea, sigma creste la {sigma_strong:.2f} pentru a 'explica' aceasta")
print(f"   discrepanta - modelul necesita variabilitate mai mare pentru a fi consistent.")
