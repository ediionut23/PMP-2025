
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


CSV_PATH = "date_promovare_examen.csv"

df = pd.read_csv(CSV_PATH)

#   Ore_Studiu, Ore_Somn, Promovare (0/1)
print("Coloane:", df.columns.tolist())
print(df.head())

# -------------------------
# (a) Balansare date
# -------------------------
y = df["Promovare"].astype(int).values
counts = pd.Series(y).value_counts().sort_index()
proportions = (counts / counts.sum())

print("\n(a) Distribuția clasei Promovare:")
print(counts)
print("Proporții:", proportions.to_dict())

# Dacă numărul de 0 și 1 este aproximativ egal (ex: 250/250), datele sunt balansate.
#   Promovare=0: 250, Promovare=1: 250  -> PERFECT BALANSAT (50% / 50%).

plt.figure()
counts.plot(kind="bar")
plt.title("Balansare clase: Promovare")
plt.xlabel("Clasă (0=nepromovat, 1=promovat)")
plt.ylabel("Număr observații")
plt.tight_layout()
plt.show()

# -------------------------
# Pregătire X + standardizare (recomandată)
# -------------------------
X = df[["Ore_Studiu", "Ore_Somn"]].values.astype(float)

# Standardizare: z = (x - mean) / std
X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
X_s = (X - X_mean) / X_std

# -------------------------
# (a) Model logistic Bayes în PyMC (ca în curs)
# -------------------------
coords = {"predictors": ["Ore_Studiu_z", "Ore_Somn_z"]}

with pm.Model(coords=coords) as model:
    alpha = pm.Normal("alpha", mu=0, sigma=10)                # intercept
    beta = pm.Normal("beta", mu=0, sigma=2, dims="predictors")# coeficienți

    mu = alpha + pm.math.dot(X_s, beta)                       # predictor liniar
    theta = pm.Deterministic("theta", pm.math.sigmoid(mu))    # p(y=1|x)

    yl = pm.Bernoulli("yl", p=theta, observed=y)              # likelihood

    idata = pm.sample(
        draws=2000,
        tune=2000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
        return_inferencedata=True
    )

print("\nPosterior summary (alpha, beta):")
print(az.summary(idata, var_names=["alpha", "beta"], round_to=3))


# Granița de decizie logistică: P(y=1|x)=0.5  <=> sigmoid(mu)=0.5 <=> mu=0
# mu = alpha + beta1*x1z + beta2*x2z = 0
# => x2z = -alpha/beta2 - (beta1/beta2)*x1z
#
# "În medie": folosim media a posteriori pentru alpha, beta1, beta2.

alpha_mean = idata.posterior["alpha"].mean(("chain", "draw")).values.item()
beta_mean = idata.posterior["beta"].mean(("chain", "draw")).values  # (2,)

b1_mean, b2_mean = beta_mean[0], beta_mean[1]

print("\n(b) Parametri medii posterior:")
print("alpha_mean =", alpha_mean)
print("beta_mean  =", beta_mean, " -> (beta_studiu_z, beta_somn_z)")

# Construim frontieră "medie" în coordonate originale (Ore_Studiu, Ore_Somn)
x1 = df["Ore_Studiu"].values
x2 = df["Ore_Somn"].values

x1_grid = np.linspace(x1.min(), x1.max(), 200)
x1z_grid = (x1_grid - X_mean[0]) / X_std[0]

# frontieră în z
eps = 1e-9
b2_safe = b2_mean if abs(b2_mean) > eps else (np.sign(b2_mean) * eps + eps)
x2z_line = (-alpha_mean / b2_safe) - ((b1_mean / b2_safe) * x1z_grid)

# convertim în original:
x2_line = x2z_line * X_std[1] + X_mean[1]

# Evaluăm separarea: probabilitatea medie posterior per observație
alpha_samps = idata.posterior["alpha"].stack(sample=("chain", "draw")).values  # (S,)
beta_samps = idata.posterior["beta"].stack(sample=("chain", "draw")).values    # (2,S)
mu_post = alpha_samps[None, :] + (X_s @ beta_samps)                            # (N,S)
p_post = 1 / (1 + np.exp(-mu_post))                                            # (N,S)
p_mean = p_post.mean(axis=1)                                                   # (N,)

y_pred = (p_mean >= 0.5).astype(int)
acc = accuracy_score(y, y_pred)
auc = roc_auc_score(y, p_mean)
cm = confusion_matrix(y, y_pred)

print("\n(b) Metrice separare (pe datasetul dat):")
print("Accuracy =", acc)
print("AUC      =", auc)
print("Confusion matrix:\n", cm)

# - Granița de decizie "în medie" este linia:
#     alpha_mean + beta1_mean*x1z + beta2_mean*x2z = 0


# Plot: puncte + frontieră medie
plt.figure(figsize=(8, 6))
plt.scatter(x1[y == 0], x2[y == 0], alpha=0.7, label="Promovare=0")
plt.scatter(x1[y == 1], x2[y == 1], alpha=0.7, label="Promovare=1")
plt.plot(x1_grid, x2_line, linewidth=2, label="Frontieră de decizie (medie posterior)")
plt.title("Regresie logistică Bayesiană – date + graniță de decizie medie")
plt.xlabel("Ore_Studiu")
plt.ylabel("Ore_Somn")
plt.legend()
plt.tight_layout()
plt.show()


b1_samples = beta_samps[0, :]
b2_samples = beta_samps[1, :]

b1_abs_mean = np.mean(np.abs(b1_samples))
b2_abs_mean = np.mean(np.abs(b2_samples))

hdi_beta = az.hdi(idata, var_names=["beta"], hdi_prob=0.94)["beta"].values  # (2,2)

print("\n(c) Influență (coeficienți pe variabile standardizate):")
print(f"beta_studiu_z mean={b1_mean:.3f}, 94% HDI=({hdi_beta[0,0]:.3f}, {hdi_beta[0,1]:.3f}), mean|beta|={b1_abs_mean:.3f}")
print(f"beta_somn_z   mean={b2_mean:.3f}, 94% HDI=({hdi_beta[1,0]:.3f}, {hdi_beta[1,1]:.3f}), mean|beta|={b2_abs_mean:.3f}")

# - Comparăm |beta_studiu_z| cu |beta_somn_z|.
# - Variabila cu |beta| mai mare influențează mai mult promovabilitatea (în acest model).
# - Dacă (de exemplu) |beta_somn_z| > |beta_studiu_z|, atunci somnul are efect mai puternic.
#   Dacă invers, atunci studiul are efect mai puternic.
#
# Notă: semnul coeficientului arată direcția efectului:
#   beta > 0 => crește probabilitatea de promovare când variabila crește
#   beta < 0 => scade probabilitatea de promovare când variabila crește
