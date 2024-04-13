import numpy as np

def compute_gae(rewards, values, gamma, lambda_):
    next_values = np.roll(values, -1, axis=1)  # simulate next values as a shifted version of values
    deltas = rewards + gamma * next_values - values
    gae = 0.0
    advantages = []
    for delta in reversed(deltas):
        gae = delta + gamma * lambda_ * gae
        advantages.insert(0, gae)
    return np.array(advantages)

# Test parameters
gamma = 0.99  # discount factor
lambda_ = 0.95  # GAE smoothing factor
batch_size = 2
seq_len = 8

# Dummy data for testing
np.random.seed(42)  # For reproducibility
values = np.random.rand(batch_size, seq_len)
rewards = np.random.rand(batch_size, seq_len)
# next_values = np.roll(values, -1, axis=1)  # simulate next values as a shifted version of values

# Compute GAE for each sequence in the batch
for i in range(batch_size):
    advantages = compute_gae(rewards[i], values[i], next_values[i], gamma, lambda_)
    print(f"Advantages for sequence {i+1}: {advantages}")

