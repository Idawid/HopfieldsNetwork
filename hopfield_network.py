# hopfield_network.py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm


class HopfieldNetwork(object):
  def train_hebb(self, train_data):
    """Hebb's learning rule (original train_weights renamed)"""
    print("Training with Hebb's rule...")
    num_data = len(train_data)
    self.num_neuron = train_data[0].shape[0]

    # Initialize weights
    W = np.zeros((self.num_neuron, self.num_neuron))
    # Mean subtraction (centering)
    rho = np.sum([np.sum(t) for t in train_data]) / (num_data * self.num_neuron)
    # Hebb rule
    for i in tqdm(range(num_data)):
      t = train_data[i] - rho
      W += np.outer(t, t)
    # Make diagonal elements of W into 0
    np.fill_diagonal(W, 0)
    # Optionally normalize by number of patterns
    W /= num_data
    self.W = W

  def train_oja(self, train_data, learning_rate=0.01, epochs=100):
    print("Training with modified Oja's rule...")
    num_data = len(train_data)
    self.num_neuron = train_data[0].shape[0]

    # Initialize weights
    W = np.zeros((self.num_neuron, self.num_neuron))

    # Modified Oja's rule for Hopfield
    for epoch in tqdm(range(epochs)):
        for i in range(num_data):
            pattern = train_data[i]
            # Calculate output
            y = np.sign(np.dot(W, pattern))  # Use sign function for binary output

            # Update weights with Oja's rule
            for j in range(self.num_neuron):
                W[j, :] += learning_rate * (pattern[j] * pattern - y[j] * y * W[j, :])

            # Ensure symmetry
            W = (W + W.T) / 2
            # Zero diagonal
            np.fill_diagonal(W, 0)

    self.W = W
  def predict(self, data, num_iter=20, threshold=0, asyn=False):
    self.num_iter = num_iter
    self.threshold = threshold
    self.asyn = asyn

    # Copy to avoid call by reference
    copied_data = np.copy(data)

    # Define predict list
    predicted = []
    for i in tqdm(range(len(data))):
      predicted.append(self._run(copied_data[i]))
    return predicted

  def _run(self, init_s):
    if self.asyn == False:
      """
      Synchronous update
      """
      # Compute initial state energy
      s = init_s
      e = self.energy(s)

      # Iteration
      for i in range(self.num_iter):
        # Update s
        s = np.sign(self.W @ s - self.threshold)
        # Compute new state energy
        e_new = self.energy(s)

        # s is converged
        if e == e_new:
          return s
        # Update energy
        e = e_new
      return s
    else:
      """
      Asynchronous update
      """
      # Compute initial state energy
      s = init_s
      e = self.energy(s)

      # Iteration
      for i in range(self.num_iter):
        for j in range(100):
          # Select random neuron
          idx = np.random.randint(0, self.num_neuron)
          # Update s
          s[idx] = np.sign(self.W[idx].T @ s - self.threshold)

        # Compute new state energy
        e_new = self.energy(s)

        # s is converged
        if e == e_new:
          return s
        # Update energy
        e = e_new
      return s

  def energy(self, s):
    return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)

  def plot_weights(self):
    plt.figure(figsize=(6, 5))
    w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
    plt.colorbar(w_mat)
    plt.title("Network Weights")
    plt.tight_layout()
    plt.savefig("weights.png")
    plt.show()
