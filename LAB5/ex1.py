import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import networkx as nx

# a
# ============================================================================

states = ["Dificil", "Mediu", "Usor"]
n_states = 3

observations_labels = ["FB", "B", "S", "NS"]
n_observations = 4

start_probability = np.array([1/3, 1/3, 1/3])

transition_probability = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.25, 0.25],
    [0.5, 0.25, 0.25]
])

emission_probability = np.array([
    [0.1, 0.2, 0.4, 0.3],
    [0.15, 0.25, 0.5, 0.1],
    [0.2, 0.3, 0.4, 0.1]
])

model = hmm.CategoricalHMM(n_components=n_states)
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability


print(f"Probabilitati initiale: {start_probability}")
print(f"\nMatricea de tranzitie:\n{transition_probability}")
print(f"\nMatricea de observatii:\n{emission_probability}")

def draw_hmm_diagram():
    G = nx.DiGraph()
    for state in states:
        G.add_node(state)

    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if transition_probability[i][j] > 0:
                G.add_edge(from_state, to_state, weight=transition_probability[i][j])

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=3000, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    edge_labels = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True,
                          arrowsize=20, arrowstyle='->', connectionstyle='arc3,rad=0.1', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)

    plt.title("Diagrama de stari HMM")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('hmm_diagram.png', dpi=300, bbox_inches='tight')
    plt.close()

draw_hmm_diagram()
print("Diagrama de stari salvata in 'hmm_diagram.png'")

# b
# ============================================================================

observed_grades = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B"]
observed_sequence = np.array([[observations_labels.index(grade)] for grade in observed_grades])

print(f"Secventa observata: {observed_grades}")

log_probability = model.score(observed_sequence)
probability = np.exp(log_probability)
print(f"\nProbabilitatea secventei: {probability:.2e}")

# c
# ============================================================================

log_prob, state_sequence = model.decode(observed_sequence, algorithm='viterbi')
most_probable_sequence = [states[i] for i in state_sequence]

print(f"Probabilitate: {np.exp(log_prob):.2e}")
print("\nSecventa optima:")
for grade, difficulty in zip(observed_grades, most_probable_sequence):
    print(f"{grade} -> {difficulty}")
print("=" * 70)
