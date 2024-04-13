# bs = 10, seq len = 100
import numpy as np
bs = 2
seq_len = 10

np.random.seed(42)  # For reproducibility


def discount_cumsum(x, discount):
    """
    Compute discounted cumulative sums of vectors.
    input:
        vector x: [x0, x1, x2]
    output:
        [x0 + discount * x1 + discount^2 * x2, x1 + discount * x2, x2]
    """
    return np.array([sum(x[j] * (discount ** j) for j in range(i, len(x)))
                     for i in range(len(x))])


gamma = 0.95
lambda_ = 0.95
# value_np = np.random.rand(bs, seq_len)
# reward_np = np.abs(np.random.rand(bs, seq_len)) / 2

values = np.random.rand(bs, seq_len)
rewards = np.random.rand(bs, seq_len)

# Append a zero bootstrap value at the end of each sequence in the batch
# This ensures that values_plus has an extra time step at the end for bootstrap
values_plus = np.concatenate([values, np.zeros((values.shape[0], 1))], axis=1)

# Calculate TD Residuals using broadcasting
deltas = rewards + gamma * values_plus[:, 1:] - values_plus[:, :-1]
advantages = discount_cumsum(deltas, 0.95 * 0.95)

print(advantages)