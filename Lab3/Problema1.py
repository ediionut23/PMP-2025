from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt

# S -> O (Spam influențează Offer)
# S -> L (Spam influențează Links)
# S -> M (Spam influențează Message length)
# L -> M (Links influențează Message length)
email_model = BayesianNetwork([('S', 'O'), ('S', 'L'), ('S', 'M'), ('L', 'M')])

# P(S) - Probabilitatea ca email-ul să fie spam
# P(S=0) = 0.6, P(S=1) = 0.4
cpd_s = TabularCPD(variable='S', variable_card=2,
                   values=[[0.6],  # S=0 (non-spam)
                           [0.4]]) # S=1 (spam)

# P(O|S) - Probabilitatea ca email-ul să conțină "offer" dat spam/non-spam
# P(O=1|S=0) = 0.1, P(O=1|S=1) = 0.7
cpd_o = TabularCPD(variable='O', variable_card=2,
                   values=[[0.9, 0.3],  # O=0: P(O=0|S=0)=0.9, P(O=0|S=1)=0.3
                           [0.1, 0.7]], # O=1: P(O=1|S=0)=0.1, P(O=1|S=1)=0.7
                   evidence=['S'],
                   evidence_card=[2])

# P(L|S) - Probabilitatea ca email-ul să conțină link-uri dat spam/non-spam
# P(L=1|S=0) = 0.3, P(L=1|S=1) = 0.8
cpd_l = TabularCPD(variable='L', variable_card=2,
                   values=[[0.7, 0.2],  # L=0: P(L=0|S=0)=0.7, P(L=0|S=1)=0.2
                           [0.3, 0.8]], # L=1: P(L=1|S=0)=0.3, P(L=1|S=1)=0.8
                   evidence=['S'],
                   evidence_card=[2])

# P(M|S,L) - Probabilitatea ca email-ul să fie lung dat spam/non-spam și link-uri
# Coloanele sunt în ordine: S=0,L=0 | S=0,L=1 | S=1,L=0 | S=1,L=1
# P(M=1|S=0,L=0) = 0.2
# P(M=1|S=0,L=1) = 0.6
# P(M=1|S=1,L=0) = 0.5
# P(M=1|S=1,L=1) = 0.9
cpd_m = TabularCPD(variable='M', variable_card=2,
                   values=[[0.8, 0.4, 0.5, 0.1],  # M=0
                           [0.2, 0.6, 0.5, 0.9]], # M=1
                   evidence=['S', 'L'],
                   evidence_card=[2, 2])

email_model.add_cpds(cpd_s, cpd_o, cpd_l, cpd_m)


print("\nP(S) - Probabilitatea ca email-ul să fie spam:")
print(cpd_s)
print("\n P(O|S) - Probabilitatea ca email-ul să conțină 'offer':")
print(cpd_o)
print("\nP(L|S) - Probabilitatea ca email-ul să conțină link-uri:")
print(cpd_l)
print("\n P(M|S,L) - Probabilitatea ca email-ul să fie lung:")
print(cpd_m)


print("a) INDEPENDENȚE ÎN REȚEA")
independencies = email_model.local_independencies(['S', 'O', 'L', 'M'])
print(independencies)

print("  • O ⊥ L, M | S  : 'Offer' este independent de 'Links' și 'Message length' dat 'Spam'")
print("  • L ⊥ O | S     : 'Links' este independent de 'Offer' dat 'Spam'")
print("  • M ⊥ O | S, L  : 'Message length' este independent de 'Offer' dat 'Spam' și 'Links'")


print("\nb) CLASIFICAREA EMAIL-URILOR")
infer = VariableElimination(email_model)

test_cases = [
    {'O': 0, 'L': 0, 'M': 0},
    {'O': 0, 'L': 0, 'M': 1},
    {'O': 0, 'L': 1, 'M': 0},
    {'O': 0, 'L': 1, 'M': 1},
    {'O': 1, 'L': 0, 'M': 0},
    {'O': 1, 'L': 0, 'M': 1},
    {'O': 1, 'L': 1, 'M': 0},
    {'O': 1, 'L': 1, 'M': 1},
]

for evidence in test_cases:
    posterior = infer.query(variables=['S'], evidence=evidence)
    prob_spam = posterior.values[1]
    classification = "SPAM" if prob_spam > 0.5 else "NON-SPAM"
    print(f"O={evidence['O']}, L={evidence['L']}, M={evidence['M']}: {classification} (P(S=1)={prob_spam:.4f})")