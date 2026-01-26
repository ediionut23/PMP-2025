import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# exercitiul 1
az.style.use('arviz-darkgrid')
np.random.seed(42)

#citire date
df = pd.read_csv('bike_daily.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# temp_c vs rentals
axes[0, 0].scatter(df['temp_c'], df['rentals'], alpha=0.5, s=20)
axes[0, 0].set_xlabel('Temperatura (°C)')
axes[0, 0].set_ylabel('Număr închirieri')
axes[0, 0].set_title('Temperatură vs Închirieri')

# humidity vs rentals
axes[0, 1].scatter(df['humidity'], df['rentals'], alpha=0.5, s=20, color='orange')
axes[0, 1].set_xlabel('Umiditate')
axes[0, 1].set_ylabel('Număr închirieri')
axes[0, 1].set_title('Umiditate vs Închirieri')

# wind_kph vs rentals
axes[1, 0].scatter(df['wind_kph'], df['rentals'], alpha=0.5, s=20, color='green')
axes[1, 0].set_xlabel('Viteza vântului (km/h)')
axes[1, 0].set_ylabel('Număr închirieri')
axes[1, 0].set_title('Vânt vs Închirieri')

season_order = ['winter', 'spring', 'summer', 'autumn']
df_plot = df.copy()
df_plot['season'] = pd.Categorical(df_plot['season'], categories=season_order, ordered=True)
df_plot.boxplot(column='rentals', by='season', ax=axes[1, 1])
axes[1, 1].set_xlabel('Anotimp')
axes[1, 1].set_ylabel('Număr închirieri')
axes[1, 1].set_title('Închirieri per anotimp')
plt.suptitle('')

plt.tight_layout()
plt.savefig('1_explorare_date.png', dpi=150)
plt.show()

print("\n" + "-" * 50)
print("OBSERVAȚII DESPRE NELINIARITĂȚI:")
print("-" * 50)
print("""
1. temp_c vs rentals: Se observă o relație pozitivă, dar cu posibilă
   neliniaritate - închirierile cresc cu temperatura până la un punct,
   apoi par să se stabilizeze sau scadă ușor la temperaturi foarte ridicate.
   Un termen polinomial (temp_c²) ar putea captura această curbură.

2. humidity vs rentals: Relație negativă slabă - umiditatea ridicată
   pare să reducă ușor numărul de închirieri.

3. wind_kph vs rentals: Relație negativă slabă - vântul puternic descurajează
   închirierile, dar efectul pare liniar.

4. season: Vara și toamna au cele mai multe închirieri, iarna cele mai putine inchirieri fata de restul
      perioadelor.
""")

## 2a) standardizarea predictorilor continui
temp_c = df['temp_c'].values
humidity = df['humidity'].values
wind_kph = df['wind_kph'].values
rentals = df['rentals'].values
is_holiday = df['is_holiday'].values

temp_mean, temp_std = temp_c.mean(), temp_c.std()
humidity_mean, humidity_std = humidity.mean(), humidity.std()
wind_mean, wind_std = wind_kph.mean(), wind_kph.std()
rentals_mean, rentals_std = rentals.mean(), rentals.std()

# Standardizare
temp_s = (temp_c - temp_mean) / temp_std
humidity_s = (humidity - humidity_mean) / humidity_std
wind_s = (wind_kph - wind_mean) / wind_std
rentals_s = (rentals - rentals_mean) / rentals_std

# temp_c² standardizat
temp_c_sq = temp_c ** 2
temp_sq_mean, temp_sq_std = temp_c_sq.mean(), temp_c_sq.std()
temp_sq_s = (temp_c_sq - temp_sq_mean) / temp_sq_std

# Codificare season
le = LabelEncoder()
season_encoded = le.fit_transform(df['season'])
season_names = le.classes_

## 2b) Model bayesian cu PyMC

season_dummies = pd.get_dummies(df['season'], drop_first=False)
season_spring = season_dummies['spring'].values.astype(float)
season_summer = season_dummies['summer'].values.astype(float)
season_autumn = season_dummies['autumn'].values.astype(float)

print(f"\nStatistici dupa standardizare: \n")
print(f"temp_s: mean={temp_s.mean():.4f}, std={temp_s.std():.4f}")
print(f"humidity_s: mean={humidity_s.mean():.4f}, std={humidity_s.std():.4f}")
print(f"wind_s: mean={wind_s.mean():.4f}, std={wind_s.std():.4f}")
print(f"rentals_s: mean={rentals_s.mean():.4f}, std={rentals_s.std():.4f}")
#variabili de predictori pentru modelele liniar si polinomial
X_linear = np.column_stack([temp_s, humidity_s, wind_s, is_holiday,
                            season_spring, season_summer, season_autumn])
predictor_names_linear = ['temp_c', 'humidity', 'wind_kph', 'is_holiday',
                          'spring', 'summer', 'autumn']
#aici pentru modelul polinomial
X_poly = np.column_stack([temp_s, temp_sq_s, humidity_s, wind_s, is_holiday,
                          season_spring, season_summer, season_autumn])
predictor_names_poly = ['temp_c', 'temp_c²', 'humidity', 'wind_kph', 'is_holiday',
                        'spring', 'summer', 'autumn']

print(f"\nMatricea X (liniar): {X_linear.shape}")
print(f"Matricea X (polinomial): {X_poly.shape}")

#aici am modelul liniar 

with pm.Model() as model_linear:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=X_linear.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + pm.math.dot(X_linear, beta)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=rentals_s)

    idata_linear = pm.sample(2000, tune=1000, chains=4,
                             target_accept=0.9, random_seed=42,
                             return_inferencedata=True)
    
print(az.summary(idata_linear, var_names=['alpha', 'beta', 'sigma']))

#2c) Model polinomial
with pm.Model() as model_poly:
    alpha = pm.Normal('alpha', mu=0, sigma=1)
    beta = pm.Normal('beta', mu=0, sigma=1, shape=X_poly.shape[1])
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + pm.math.dot(X_poly, beta)
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=rentals_s)

    idata_poly = pm.sample(2000, tune=1000, chains=4,
                           target_accept=0.9, random_seed=42,
                           return_inferencedata=True)

print(az.summary(idata_poly, var_names=['alpha', 'beta', 'sigma']))


#ex E
# Inference and Diagnostics:
# Diagnostice convergență
print("\nDiagnostice model liniar:")
print(f"R-hat max: {az.rhat(idata_linear).max()}")
print(f"ESS min: {az.ess(idata_linear).min()}")

print("\nDiagnostice model polinomial:")
print(f"R-hat max: {az.rhat(idata_poly).max()}")
print(f"ESS min: {az.ess(idata_poly).min()}")

#analiza model polinomial 
az.plot_trace(idata_poly, var_names=['alpha', 'beta', 'sigma'])
plt.tight_layout()
plt.savefig('3_trace_plot_poly.png', dpi=150)
plt.show()

#analiza influenta variabilelor
# Model liniar
beta_linear_mean = idata_linear.posterior['beta'].mean(('chain', 'draw')).values
beta_linear_std = idata_linear.posterior['beta'].std(('chain', 'draw')).values

print("\nModel Liniar - Coeficienți (standardizate):")
for i, name in enumerate(predictor_names_linear):
    print(f"  {name}: β = {beta_linear_mean[i]:.4f} ± {beta_linear_std[i]:.4f}")

# Model polinomial
beta_poly_mean = idata_poly.posterior['beta'].mean(('chain', 'draw')).values
beta_poly_std = idata_poly.posterior['beta'].std(('chain', 'draw')).values

print("\nModel Polinomial - Coeficienți (pe date standardizate):")
for i, name in enumerate(predictor_names_poly):
    print(f"  {name}: β = {beta_poly_mean[i]:.4f} ± {beta_poly_std[i]:.4f}")

#(|β| maxim)
max_idx_linear = np.argmax(np.abs(beta_linear_mean))
max_idx_poly = np.argmax(np.abs(beta_poly_mean))
print(f"\nVariabila cu cea mai mare influență (model liniar): {predictor_names_linear[max_idx_linear]} (|β|={np.abs(beta_linear_mean[max_idx_linear]):.4f})")
print(f"Variabila cu cea mai mare influență (model polinomial): {predictor_names_poly[max_idx_poly]} (|β|={np.abs(beta_poly_mean[max_idx_poly]):.4f})")


fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Model liniar
az.plot_forest(idata_linear, var_names=['beta'], combined=True, ax=axes[0])
axes[0].set_title('Model Liniar - Coeficienți')
axes[0].set_yticklabels(predictor_names_linear[::-1])

# Model polinomial
az.plot_forest(idata_poly, var_names=['beta'], combined=True, ax=axes[1])
axes[1].set_title('Model Polinomial - Coeficienți')
axes[1].set_yticklabels(predictor_names_poly[::-1])

plt.tight_layout()
plt.savefig('3_forest_plot_coef.png', dpi=150)
plt.show()

#Exercitiul 4 

# 4a) Compararea modelelor via WAIC și LOO

pm.compute_log_likelihood(idata_linear, model=model_linear)
pm.compute_log_likelihood(idata_poly, model=model_poly)
# WAIC
waic_linear = az.waic(idata_linear)
waic_poly = az.waic(idata_poly)
print("\n4.a) Comparare WAIC:")
print(f"Model Liniar WAIC: {waic_linear.elpd_waic:.2f} ± {waic_linear.se:.2f}")
print(f"Model Polinomial WAIC: {waic_poly.elpd_waic:.2f} ± {waic_poly.se:.2f}")
# LOO
loo_linear = az.loo(idata_linear)
loo_poly = az.loo(idata_poly)
print("\nComparare LOO:")
print(f"Model Liniar LOO: {loo_linear.elpd_loo:.2f} ± {loo_linear.se:.2f}")
print(f"Model Polinomial LOO: {loo_poly.elpd_loo:.2f} ± {loo_poly.se:.2f}")
compare_df = az.compare({
    'Linear': idata_linear,
    'Polynomial': idata_poly
}, ic='waic')

print("\nTabel comparație modele:")
print(compare_df)

print("JUSTIFICARE ALEGERE MODEL:")
print("-" * 50)
if waic_poly.elpd_waic > waic_linear.elpd_waic:
    print("""
Modelul POLINOMIAL este preferat deoarece:
- Are ELPD-WAIC mai mare (mai aproape de 0 = mai bun)
- Termenul temp_c² captează relația neliniară dintre temperatură și închirieri
- Diferența de WAIC justifică complexitatea adăugată
""")
else:
    print("""
Modelul LINIAR este preferat deoarece:
- Are ELPD-WAIC mai mare sau similar
- Modelul mai simplu este preferat când performanța este comparabilă
""")
    
# 4.b) Posterior Predictive Checks(PPC)
with model_poly:
    pm.sample_posterior_predictive(idata_poly, extend_inferencedata=True, random_seed=42)
#Posterior Predictive Check plot 
az.plot_ppc(idata_poly, num_pp_samples=100)
plt.title('Posterior Predictive Check - Model Polinomial')
plt.tight_layout()
plt.savefig('4b_ppc_poly.png', dpi=150)
plt.show()

# Vizualizare predicție medie și incertitudine vs temp_c
temp_grid = np.linspace(temp_c.min(), temp_c.max(), 100)
temp_grid_s = (temp_grid - temp_mean) / temp_std
temp_grid_sq_s = ((temp_grid ** 2) - temp_sq_mean) / temp_sq_std
# Folosim valorile medii
humidity_mean_s = 0  
wind_mean_s = 0
is_holiday_mean = 0.5
spring_mean = 0.25
summer_mean = 0.25
autumn_mean = 0.25

X_pred = np.column_stack([
    temp_grid_s,
    temp_grid_sq_s,
    np.full_like(temp_grid, humidity_mean_s),
    np.full_like(temp_grid, wind_mean_s),
    np.full_like(temp_grid, is_holiday_mean),
    np.full_like(temp_grid, spring_mean),
    np.full_like(temp_grid, summer_mean),
    np.full_like(temp_grid, autumn_mean)
])

# Calculăm predicțiile
alpha_samples = idata_poly.posterior['alpha'].stack(sample=('chain', 'draw')).values
beta_samples = idata_poly.posterior['beta'].stack(sample=('chain', 'draw')).values
# mu = alpha + X @ beta
mu_pred = alpha_samples[None, :] + X_pred @ beta_samples 
# scala originală
mu_pred_original = mu_pred * rentals_std + rentals_mean
# media și intervalele HDI
mu_mean = mu_pred_original.mean(axis=1)
mu_hdi = np.percentile(mu_pred_original, [2.5, 97.5], axis=1)
# un plot dragut cu predicția medie și incertitudinea
plt.figure(figsize=(10, 6))
plt.scatter(temp_c, rentals, alpha=0.3, s=20, label='Date observate')
plt.plot(temp_grid, mu_mean, 'r-', linewidth=2, label='Media predicției')
plt.fill_between(temp_grid, mu_hdi[0], mu_hdi[1], alpha=0.3, color='red',
                 label='95% HDI')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Număr închirieri')
plt.title('Predicție medie și incertitudine vs Temperatură (Model Polinomial)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('4b_prediction_vs_temp.png', dpi=150)
plt.show()

# Exercitiul 5
# Calculăm percentila 75 pe scala originală
Q75 = np.percentile(rentals, 75)
print(f"Percentila 75 (Q): {Q75:.0f} închirieri")
# Creăm variabila binară
is_high_demand = (rentals >= Q75).astype(int)

print(f"\nDistribuție is_high_demand:")
print(f"  Low demand (0): {(is_high_demand == 0).sum()} ({100*(is_high_demand == 0).mean():.1f}%)")
print(f"  High demand (1): {(is_high_demand == 1).sum()} ({100*(is_high_demand == 1).mean():.1f}%)")
#plot dragut pentru distributia noii variabile
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(rentals, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(Q75, color='red', linestyle='--', linewidth=2, label=f'Q75 = {Q75:.0f}')
plt.xlabel('Număr închirieri')
plt.ylabel('Frecvență')
plt.title('Distribuția închirierilor cu pragul Q75')
plt.legend()
plt.subplot(1, 2, 2)
plt.bar(['Low demand', 'High demand'],
        [(is_high_demand == 0).sum(), (is_high_demand == 1).sum()],
        color=['steelblue', 'coral'], edgecolor='black')
plt.ylabel('Număr observații')
plt.title('Distribuția cererii')
plt.tight_layout()
plt.savefig('5_high_demand_distribution.png', dpi=150)
plt.show()

#Exercitiul 6
# Regresia logistica bayesiana cu PyMC
# Justificare pentru includerea temp_c²:
print("\nJustificare pentru includerea/excluderea temp_c²:")
print("""
Vom PĂSTRA termenul temp_c² deoarece:
1. Relația dintre temperatură și închirieri pare neliniară (observată la explorare)
2. Modelul polinomial a performat mai bine la comparația WAIC
3. Pentru predicția cererii ridicate, capturarea efectului de "saturație"
   la temperaturi extreme poate fi importantă
""")
#folosim aceeasi matrice
X_logistic = X_poly.copy()
predictor_names_logistic = predictor_names_poly.copy()
# Model de regresie logistică
with pm.Model() as model_logistic:
    alpha = pm.Normal('alpha', mu=0, sigma=2)
    beta = pm.Normal('beta', mu=0, sigma=2, shape=X_logistic.shape[1])

    mu = alpha + pm.math.dot(X_logistic, beta)
    theta = pm.Deterministic('theta', pm.math.sigmoid(mu))
    y_obs = pm.Bernoulli('y_obs', p=theta, observed=is_high_demand)
    idata_logistic = pm.sample(2000, tune=2000, chains=4,
                               target_accept=0.95, random_seed=42,
                               return_inferencedata=True)

print("\nRezumat model logistic:")
print(az.summary(idata_logistic, var_names=['alpha', 'beta']))


#Exercitiul 7
# Evaluarea performanței modelului logistic
#HDI 95%
hdi_alpha = az.hdi(idata_logistic, var_names=['alpha'], hdi_prob=0.95)
hdi_beta = az.hdi(idata_logistic, var_names=['beta'], hdi_prob=0.95)
print("\nHDI 95% pentru parametri:")
alpha_mean_log = idata_logistic.posterior['alpha'].mean(('chain', 'draw')).values
print(f"alpha: {alpha_mean_log:.4f}, HDI 95%: [{hdi_alpha['alpha'].values[0]:.4f}, {hdi_alpha['alpha'].values[1]:.4f}]")

beta_mean_log = idata_logistic.posterior['beta'].mean(('chain', 'draw')).values
beta_hdi = hdi_beta['beta'].values

print("\nCoeficienți beta și HDI 95%:")
for i, name in enumerate(predictor_names_logistic):
    print(f"  {name}: β = {beta_mean_log[i]:.4f}, HDI 95%: [{beta_hdi[i, 0]:.4f}, {beta_hdi[i, 1]:.4f}]")
# Folosim |beta| mediu ca măsură a influenței
abs_beta = np.abs(beta_mean_log)
max_influence_idx = np.argmax(abs_beta)

print("ANALIZA INFLUENȚEI ASUPRA PROBABILITĂȚII DE CERERE RIDICATĂ:")
# Sortăm după influență
sorted_idx = np.argsort(abs_beta)[::-1]
print("\nVariabile ordonate după influență (|β|):")
for idx in sorted_idx:
    sign = "+" if beta_mean_log[idx] > 0 else "-"
    print(f"  {predictor_names_logistic[idx]}: |β| = {abs_beta[idx]:.4f} ({sign})")
print(f"\n>>> Variabila cu cea mai mare influență: {predictor_names_logistic[max_influence_idx]}")
print(f"    β = {beta_mean_log[max_influence_idx]:.4f}")

# Interpretare
print("INTERPRETARE:")
print(f"""
Variabila '{predictor_names_logistic[max_influence_idx]}' influențează cel mai mult
probabilitatea de cerere ridicată (is_high_demand).

Interpretarea coeficienților (pe date standardizate):
- β > 0: creșterea variabilei crește probabilitatea de cerere ridicată
- β < 0: creșterea variabilei scade probabilitatea de cerere ridicată

Efectul temperaturii:
- temp_c (liniar): β = {beta_mean_log[0]:.4f}
- temp_c² (pătratic): β = {beta_mean_log[1]:.4f}
  Împreună, acestea sugerează o relație neliniară cu un optim de temperatură.

Efectul umidității: β = {beta_mean_log[2]:.4f}
  Umiditatea ridicată {'scade' if beta_mean_log[2] < 0 else 'crește'} probabilitatea de cerere ridicată.

Efectul vântului: β = {beta_mean_log[3]:.4f}
  Vântul puternic {'scade' if beta_mean_log[3] < 0 else 'crește'} probabilitatea de cerere ridicată.
""")

# Plot dragut forest pentru coeficienții logistici
fig, ax = plt.subplots(figsize=(10, 6))
az.plot_forest(idata_logistic, var_names=['beta'], combined=True, ax=ax,
               hdi_prob=0.95)
ax.set_title('Coeficienți model logistic (HDI 95%)')
ax.set_yticklabels(predictor_names_logistic[::-1])
ax.axvline(0, color='black', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('7_logistic_coefficients_hdi.png', dpi=150)
plt.show()

# Plot dragut posterior pentru coeficienții importanți
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()
for i, name in enumerate(predictor_names_logistic):
    az.plot_posterior(idata_logistic, var_names=['beta'],
                      coords={'beta_dim_0': i}, ax=axes[i],
                      hdi_prob=0.95)
    axes[i].set_title(f'β_{name}')
plt.tight_layout()
plt.savefig('7_posterior_distributions.png', dpi=150)
plt.show()
