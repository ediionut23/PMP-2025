import numpy as np
import matplotlib.pyplot as plt

n_simulations = 1000
lambdas = [1, 2, 5, 10]

X1 = np.random.poisson(lam=1, size=n_simulations)
X2 = np.random.poisson(lam=2, size=n_simulations)
X3 = np.random.poisson(lam=5, size=n_simulations)
X4 = np.random.poisson(lam=10, size=n_simulations)

X_mixed = []
for i in range(n_simulations):
    chosen_lambda = np.random.choice(lambdas)
    value = np.random.poisson(lam=chosen_lambda)
    X_mixed.append(value)

X_mixed = np.array(X_mixed)

plt.hist(X1, bins=15, color='orange', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (λ=1)')
plt.xlabel('Număr de apeluri')
plt.ylabel('Frecvență')
plt.show()

plt.hist(X2, bins=15, color='green', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (λ=2)')
plt.xlabel('Număr de apeluri')
plt.ylabel('Frecvență')
plt.show()

plt.hist(X3, bins=15, color='blue', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (λ=5)')
plt.xlabel('Număr de apeluri')
plt.ylabel('Frecvență')
plt.show()

plt.hist(X4, bins=15, color='red', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson (λ=10)')
plt.xlabel('Număr de apeluri')
plt.ylabel('Frecvență')
plt.show()

plt.hist(X_mixed, bins=20, color='purple', edgecolor='black', alpha=0.7)
plt.title('Distribuția Poisson Randomizată (λ aleatoriu)')
plt.xlabel('Număr de apeluri')
plt.ylabel('Frecvență')
plt.show()


# b) distributia randomizata nu are neaparat un varf, tinde mai mult sa aiba un platou,
# in functie de cate valori diferite de lambda sunt alese si cat de des apar acestea in selectie.
#la valorile fixe de lambda, se observa un varf clar la valoarea medie a distributiei Poisson,
# in timp ce la distributia randomizata, varful este mai putin evident si
#deci in lumea reala, daca rata de aparitie a evenimentelor variaza semnificativ,
# modelul cu lambda randomizat poate oferi o reprezentare mai realista a datelor