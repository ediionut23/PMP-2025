import numpy as np
from hmmlearn import hmm


states = ["W", "R", "S"]
obs_map = {"L": 0, "M": 1, "H": 2}


start_probability = np.array([0.4, 0.3, 0.3])

transition_probability = np.array([
    [0.6, 0.3, 0.1],  
    [0.2, 0.7, 0.1],  
    [0.3, 0.2, 0.5],  
])

emission_probability = np.array([
    [0.1, 0.7, 0.2],  
    [0.05, 0.25, 0.7],  
    [0.8, 0.15, 0.05],  
])

model = hmm.CategoricalHMM(n_components=3, init_params="")
model.startprob_ = start_probability
model.transmat_ = transition_probability
model.emissionprob_ = emission_probability


obs_seq = np.array([obs_map[o] for o in "M H L".split()]).reshape(-1, 1)


hidden_states = model.predict(obs_seq)
print("Most likely hidden states:", [states[s] for s in hidden_states])

logprob, posteriors = model.score_samples(obs_seq)

print("\nPosterior probabilities (gamma):")
print(posteriors)

print("\nlog P(M,H,L) =", logprob)
print("P(M,H,L) =", np.exp(logprob))
