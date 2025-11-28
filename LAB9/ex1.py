import pymc as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt

Y_values = [0, 5, 10]
theta_values = [0.2, 0.5]

posterior_results = {}
models = {}  

# (a) POSTERIORUL PENTRU n — toate combinațiile (Y, θ)
fig, axes = plt.subplots(len(Y_values), len(theta_values), figsize=(10, 8))

for i, Y in enumerate(Y_values):
    for j, theta in enumerate(theta_values):

        with pm.Model() as model:

            n = pm.Poisson("n", mu=10)
            Y_obs = pm.Binomial("Y_obs", n=n, p=theta, observed=Y)

            idata = pm.sample(
                draws=2000,
                tune=2000,
                chains=2,
                cores=1,
                random_seed=2025,
                progressbar=False,
                step=pm.Metropolis(),
                return_inferencedata=True
            )

        posterior_results[(Y, theta)] = idata
        models[(Y, theta)] = model

        az.plot_posterior(idata, var_names=["n"], ax=axes[i, j])
        axes[i, j].set_title(f"Posterior n | Y={Y}, θ={theta}")

plt.tight_layout()
plt.show()

# (b)
"""
1. Y mare → n mare (trebuie mulți clienți pentru a obține multe cumpărări)
2. Y mic → n mic
3. θ mic → n trebuie să fie mai mare pentru a produce același Y
4. θ mare → n poate fi mai mic

Concluzie:
    - Y determină „nivelul” posteriorului pentru n.
    - θ determină cât de mult trebuie ajustat n pentru a explica datele.
"""



predictive_results = {}

for Y in Y_values:
    for theta in theta_values:

        idata = posterior_results[(Y, theta)]
        model = models[(Y, theta)]

        idata_pp = pm.sample_posterior_predictive(
            idata,
            model=model,
            extend_inferencedata=True
        )

        predictive_results[(Y, theta)] = idata_pp

        az.plot_ppc(
            idata_pp,
            figsize=(6, 4),
            mean=False
        )
        plt.title(f"PPC Y* | Y={Y}, θ={theta}")
        plt.tight_layout()
        plt.show()


# (d) Diferența între posterior(n) și PPC
"""
Posterior(n):
    - ne spune ce credem despre numărul total de clienți în ziua OBSERVATĂ.
    - distribuție asupra unei variabile latente din model.

Predictive posterior (PPC):
    - ne spune ce valori viitoare Y* sunt probabile,
      ținând cont de posterior(n).
    - combină:
        * incertitudinea despre n
        * variabilitatea binomială Y* | n, θ
    - este întotdeauna mai lat, pentru că include MULT mai multă incertitudine.

"""
