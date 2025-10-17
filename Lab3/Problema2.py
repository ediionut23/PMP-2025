import numpy as np
from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# a) si b) din laboratorul precedent SIMULARE
print("a) + b) SIMULARE")

np.random.seed(42)
num_simulations = 100000

red_drawn = 0

for _ in range(num_simulations):
    die = np.random.randint(1, 7)

    if die in [2, 3, 5]: 
        red, blue, black = 3, 4, 3
    elif die == 6: 
        red, blue, black = 4, 4, 2
    else: 
        red, blue, black = 3, 5, 2

    total = red + blue + black

    ball = np.random.choice(['red', 'blue', 'black'],
                           p=[red/total, blue/total, black/total])

    if ball == 'red':
        red_drawn += 1

prob_simulation = red_drawn / num_simulations
print(f"P(Roșu) din simulare: {prob_simulation:.4f}\n")

print(" REȚEA BAYESIANĂ")

model = BayesianNetwork([('Die', 'Ball')])

cpd_die = TabularCPD('Die', 6, [[1/6]] * 6)
print(cpd_die)

# P(Ball|Die) pentru fiecare valoare a zarului
ball_probs = []
for die_val in range(1, 7):
    if die_val in [2, 3, 5]:  # Prim
        red, blue, black = 3, 4, 3
    elif die_val == 6:
        red, blue, black = 4, 4, 2
    else:  # 1 sau 4
        red, blue, black = 3, 5, 2

    total = red + blue + black
    ball_probs.append([red/total, blue/total, black/total])

cpd_ball = TabularCPD('Ball', 3, np.array(ball_probs).T,
                      evidence=['Die'], evidence_card=[6])
print(cpd_ball)

model.add_cpds(cpd_die, cpd_ball)

infer = VariableElimination(model)
result = infer.query(['Ball'])

prob_bayesian = result.values[0]
print(f"P(Roșu) din Bayesian Network: {prob_bayesian:.4f}\n")

# Comparație
print(f"Simulare: {prob_simulation:.4f}")
print(f"Bayesian: {prob_bayesian:.4f}")
