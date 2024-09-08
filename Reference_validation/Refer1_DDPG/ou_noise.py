import numpy as np
import random
import copy 

class OUNoise():
    """ Ornstein-Uhlenbeck process """

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2, sigma_min=0.01, sigma_decay=0.99):
        """ Initialize parameters and noise process """
        self.mu = mu * np.ones(size)
        self.theta = theta 
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay
        self.seed = random.seed(seed)
        self.size = size
        self.reset()

    def reset(self):
        """ Reset the interal state (= noise) to mean (mu). """
        self.state = copy.copy(self.mu)
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)

    def sample(self):
        """ Update internal state and return it as a noise sample """
        x = self.state 
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(self.size)
        self.state = x + dx
        return self.state