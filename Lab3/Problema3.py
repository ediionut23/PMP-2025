import numpy as np
import math
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

np.random.seed(42)
wins = [0, 0]  # wins[0] = P0, wins[1] = P1

for _ in range(10000):
    starter = np.random.randint(0, 2)  #se arunca moneda
    n = np.random.randint(1, 7)  #se arunca zarul

    # Celălalt jucător aruncă moneda de 2n ori
    # P0 are monedă corectă (p=0.5), P1 are monedă trucată (p=4/7)
    other_player = 1 - starter
    p = 0.5 if other_player == 0 else 4/7
    m = np.random.binomial(2*n, p)

    # Jucătorul din prima rundă câștigă dacă n >= m
    winner = starter if n >= m else other_player
    wins[winner] += 1

print(f"  P0 câștigă: {wins[0]} jocuri ({wins[0]/10000:.2%})")
print(f"  P1 câștigă: {wins[1]} jocuri ({wins[1]/10000:.2%})")

print("b)")

# Structura: Starter -> N, Starter -> M, N -> M
game_model = BayesianNetwork([('Starter', 'N'), ('Starter', 'M'), ('N', 'M')])

# P(Starter) - Moneda corectă decide cine începe
cpd_starter = TabularCPD('Starter', 2, [[0.5], [0.5]])

# P(N|Starter) - Zarul este corect pentru ambii jucători
cpd_n = TabularCPD('N', 6, [[1/6]*2]*6,
                   evidence=['Starter'], evidence_card=[2])

# P(M|Starter, N) - Distribuție binomială pentru numărul de capete
# M poate fi între 0 și 12 (maxim când N=6, aruncăm 2*6=12 monede)
m_vals = []
for s in [0, 1]:  # Pentru fiecare starter
    # P0 are monedă corectă (p=0.5), P1 are monedă trucată (p=4/7)
    # Dar celălalt jucător aruncă moneda în runda 2!
    other_player = 1 - s
    p = 0.5 if other_player == 0 else 4/7

    for n in range(1, 7):  # Pentru fiecare valoare a zarului (1-6)
        col = []
        for m in range(13):  # M poate fi 0-12
            if m <= 2*n:
                # Probabilitate binomială: C(2n,m) * p^m * (1-p)^(2n-m)
                prob = math.comb(2*n, m) * (p**m) * ((1-p)**(2*n-m))
                col.append(prob)
            else:
                col.append(0)
        m_vals.append(col)

cpd_m = TabularCPD('M', 13, np.array(m_vals).T,
                   evidence=['Starter', 'N'], evidence_card=[2, 6])

# Adăugarea CPD-urilor la model
print(cpd_starter)
print(cpd_n)
print(cpd_m)

game_model.add_cpds(cpd_starter, cpd_n, cpd_m)

# Verificarea modelului
if game_model.check_model():
    print("\n Modelul Bayesian este valid!")
else:
    print("\n Eroare în model!")

print("\nc)")

infer = VariableElimination(game_model)
posterior = infer.query(['Starter'], evidence={'M': 1})

print(f"P(P0 a început | M=1) = {posterior.values[0]:.4f}")
print(f"P(P1 a început | M=1) = {posterior.values[1]:.4f}")
print(f"Cel mai probabil a început: {'P0' if posterior.values[0] > posterior.values[1] else 'P1'}")
