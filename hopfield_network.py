# hopfield_network.py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm


class HopfieldNetwork(object):
  def train_hebb(self, train_data):
    """Hebb's learning rule with 2D patterns"""
    print("Training with Hebb's rule...")
    num_data = len(train_data)
    pattern_shape = train_data[0].shape
    self.num_neuron = np.prod(pattern_shape)

    # Initialize weights
    W = np.zeros((self.num_neuron, self.num_neuron))

    # Mean subtraction (centering)
    rho = np.mean(train_data)

    # Hebb rule
    for i in tqdm(range(num_data)):
      t = train_data[i] - rho
      t_flat = t.flatten()
      W += np.outer(t_flat, t_flat)

    np.fill_diagonal(W, 0)
    W /= num_data
    self.W = W


  def train_oja(self, train_data, learning_rate=0.01, epochs=100):
    """Modified Oja's rule with 2D patterns"""
    print("Training with modified Oja's rule...")
    num_data = len(train_data)
    pattern_shape = train_data[0].shape
    self.num_neuron = np.prod(pattern_shape)

    W = np.zeros((self.num_neuron, self.num_neuron))

    for epoch in tqdm(range(epochs)):
      for i in range(num_data):
        pattern = train_data[i].flatten()
        y = np.sign(np.dot(W, pattern))

        for j in range(self.num_neuron):
          W[j, :] += learning_rate * (pattern[j] * pattern - y[j] * y * W[j, :])

        W = (W + W.T) / 2
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
    if not self.asyn:
      """Synchronous update"""
      s = init_s
      e = self.energy(s)

      # Iteration
      for i in range(self.num_iter):
        s_flat = s.flatten()
        s_new = np.sign(self.W @ s_flat - self.threshold)
        s = s_new.reshape(init_s.shape)

        e_new = self.energy(s)

        # s is converged
        if e == e_new:
          return s
        # Update energy
        e = e_new
      return s
    else:
      """Asynchronous update"""
      s = init_s
      e = self.energy(s)
      height, width = s.shape

      # Iteration
      for i in range(self.num_iter):
        for j in range(100):
          # Select random position in 2D
          h_idx = np.random.randint(0, height)
          w_idx = np.random.randint(0, width)
          flat_idx = h_idx * width + w_idx

          s_flat = s.flatten()
          s_flat[flat_idx] = np.sign(
            self.W[flat_idx].T @ s_flat - self.threshold)
          s = s_flat.reshape(s.shape)

        e_new = self.energy(s)

        # s is converged
        if e == e_new:
          return s
        # Update energy
        e = e_new
      return s

  def energy(self, s):
    s_flat = s.flatten()
    return -0.5 * s_flat @ self.W @ s_flat + np.sum(s_flat * self.threshold)

  def plot_weights(self):
    plt.figure(figsize=(6, 5))
    w_mat = plt.imshow(self.W, cmap=cm.coolwarm)
    plt.colorbar(w_mat)
    plt.title("Network Weights")
    plt.tight_layout()
    plt.savefig("weights.png")
    plt.show()
