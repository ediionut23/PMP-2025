import numpy as np
states = ["Dificil", "Mediu", "Usor"]
observations_labels = ["FB", "B", "S", "NS"]
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

observed_grades = ["FB", "FB", "S", "B", "B", "S", "B", "B", "NS", "B", "B"]
observed_sequence = [observations_labels.index(grade) for grade in observed_grades]
"""
    1. Initializare: delta_1(i) = pi_i * b_i(o_1)
    2. Recursie: delta_t(j) = max_i[delta_{t-1}(i) * a_ij] * b_j(o_t)
                 psi_t(j) = argmax_i[delta_{t-1}(i) * a_ij]
    3. Terminare: P* = max_i[delta_T(i)]
                  q*_T = argmax_i[delta_T(i)]
    4. Backtracking: q*_{t-1} = psi_t(q*_t)
    """
def viterbi_manual(obs_seq, start_prob, trans_prob, emis_prob, states):
    T = len(obs_seq)
    N = len(start_prob)

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    for i in range(N):
        delta[0, i] = start_prob[i] * emis_prob[i, obs_seq[0]]
        psi[0, i] = 0

    for t in range(1, T):
        for j in range(N):
            max_val = -1
            max_idx = 0

            for i in range(N):
                val = delta[t-1, i] * trans_prob[i, j]
                if val > max_val:
                    max_val = val
                    max_idx = i

            delta[t, j] = max_val * emis_prob[j, obs_seq[t]]
            psi[t, j] = max_idx

    prob = np.max(delta[T-1, :])
    last_state = np.argmax(delta[T-1, :])

    path = np.zeros(T, dtype=int)
    path[T-1] = last_state

    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return prob, path

prob_manual, path_manual = viterbi_manual(
    observed_sequence,
    start_probability,
    transition_probability,
    emission_probability,
    states
)

most_probable_sequence = [states[i] for i in path_manual]

print(f"Probabilitate secventa optima: {prob_manual:.2e}")
print("\nSecventa optima:")
for grade, difficulty in zip(observed_grades, most_probable_sequence):
    print(f"{grade} -> {difficulty}")
