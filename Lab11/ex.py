import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# --- 1. Încărcarea și Pregătirea Datelor ---
try:
    df = pd.read_csv('Prices.csv')
    print("Fisierul Prices.csv a fost incarcat cu succes.")
except FileNotFoundError:
    print("Eroare: Nu s-a gasit fisierul 'Prices.csv'. Asigura-te ca e in acelasi folder.")
    exit()

# Transformarea variabilelor
# x1 = Speed, x2 = ln(HardDrive)
price = df['Price'].values
speed = df['Speed'].values
hard_drive = df['HardDrive'].values
log_hd = np.log(hard_drive)

print(df[['Price', 'Speed', 'HardDrive']].head())
print("Log HardDrive (primele 5):", log_hd[:5])

# ==========================================
# a) Definirea Modelului și Simulare
# ==========================================
print("\n--- Incepem simularea modelului principal ---")
with pm.Model() as model_pc:
    # MODIFICARE: Folosim pm.Data FĂRĂ mutable=True
    # În versiunea ta, pm.Data creează automat un container SharedVariable
    x1_shared = pm.Data("x1_shared", speed)
    x2_shared = pm.Data("x2_shared", log_hd)
    
    # Distribuții a priori (Weakly informative)
    alpha = pm.Normal("alpha", mu=0, sigma=3000) 
    beta1 = pm.Normal("beta1", mu=0, sigma=100)    # Panta pentru Speed
    beta2 = pm.Normal("beta2", mu=0, sigma=1000)   # Panta pentru Log(HDD)
    
    sigma = pm.HalfNormal("sigma", sigma=1000)
    
    # Modelul liniar determinist
    mu = pm.Deterministic("mu", alpha + beta1 * x1_shared + beta2 * x2_shared)
    
    # Likelihood
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=price)
    
    # Simulare
    # target_accept=0.9 ajuta la stabilitatea matematica a simularii
    idata = pm.sample(2000, tune=2000, return_inferencedata=True, target_accept=0.9)

# ==========================================
# b) Estimări 95% HDI
# ==========================================
print("\n--- Rezumat a posteriori (b) ---")
summary = az.summary(idata, var_names=["beta1", "beta2", "alpha", "sigma"], hdi_prob=0.95)
print(summary)

# Forest Plot
plt.figure(figsize=(10, 5))
az.plot_forest(idata, var_names=["beta1", "beta2"], hdi_prob=0.95, combined=True)
plt.title("Estimări 95% HDI pentru beta1 și beta2")
plt.show()

# ==========================================
# c) Interpretare (Text în consolă)
# ==========================================
print("\n--- Punctul c) ---")
print("Verifica intervalele HDI de mai sus.")
print("Daca intervalul pentru beta1 sau beta2 NU contine 0, predictorul este util.")

# ==========================================
# d) Predicție Preț Mediu (mu) pentru PC: 33 MHz, 540 MB
# ==========================================
print("\n--- Punctul d) Predicție MU ---")
new_speed = [33]
new_log_hd = [np.log(540)]

with model_pc:
    # Actualizăm datele in containerul pm.Data existent
    pm.set_data({"x1_shared": new_speed, "x2_shared": new_log_hd})
    
    # Eșantionăm doar 'mu' (media)
    ppc_mu = pm.sample_posterior_predictive(idata, var_names=["mu"])

# Extragem datele (compatibilitate pentru diverse versiuni PyMC)
if hasattr(ppc_mu, "posterior_predictive"):
    mu_samples = ppc_mu.posterior_predictive["mu"].values.flatten()
else:
    mu_samples = ppc_mu["mu"].flatten()

hdi_mu = az.hdi(mu_samples, hdi_prob=0.90)
print(f"Intervalul 90% HDI pentru prețul MEDIU (mu): {hdi_mu}")

# ==========================================
# e) Predicție Preț Final (y) pentru același PC
# ==========================================
print("\n--- Punctul e) Predicție Y ---")
with model_pc:
    # Încercăm să generăm predicții (y_pred)
    # Folosim bloc try-except pentru compatibilitate între versiuni (predictions=True vs legacy)
    try:
        ppc_pred = pm.sample_posterior_predictive(idata, var_names=["y_pred"], predictions=True)
        # Accesare pentru PyMC v5 modern
        if hasattr(ppc_pred, "predictions"):
             y_samples = ppc_pred.predictions["y_pred"].values.flatten()
        else:
             y_samples = ppc_pred.posterior_predictive["y_pred"].values.flatten()
    except:
        # Fallback pentru versiuni mai vechi sau diferite
        ppc_pred = pm.sample_posterior_predictive(idata, var_names=["y_pred"])
        y_samples = ppc_pred.posterior_predictive["y_pred"].values.flatten()

hdi_y = az.hdi(y_samples, hdi_prob=0.90)
print(f"Intervalul 90% HDI pentru prețul PREZIS (y - un singur PC): {hdi_y}")

# ==========================================
# Bonus: Variabila Premium
# ==========================================
print("\n--- Bonus: Analiza Premium ---")
# 1 = yes, 0 = no
df['Premium_Binary'] = df['Premium'].apply(lambda x: 1 if x == 'yes' else 0)
premium_vals = df['Premium_Binary'].values

with pm.Model() as model_bonus:
    # Date (fără mutable=True)
    x1 = pm.Data("x1", speed)
    x2 = pm.Data("x2", log_hd)
    x3 = pm.Data("x3", premium_vals)
    
    # Priors
    alpha = pm.Normal("alpha", mu=0, sigma=3000)
    beta1 = pm.Normal("beta1", mu=0, sigma=100)
    beta2 = pm.Normal("beta2", mu=0, sigma=1000)
    beta3 = pm.Normal("beta3", mu=0, sigma=1000) # Coeficientul Premium
    sigma = pm.HalfNormal("sigma", sigma=1000)
    
    # Ecuatia
    mu = alpha + beta1 * x1 + beta2 * x2 + beta3 * x3
    
    y_pred = pm.Normal("y_pred", mu=mu, sigma=sigma, observed=price)
    
    idata_bonus = pm.sample(2000, tune=2000, return_inferencedata=True, target_accept=0.9)

print("Rezumat Bonus (vezi beta3):")
print(az.summary(idata_bonus, var_names=["beta3"], hdi_prob=0.95))

plt.figure()
az.plot_posterior(idata_bonus, var_names=["beta3"], hdi_prob=0.95, ref_val=0)
plt.title("Impactul Premium (Beta3)")
plt.show()