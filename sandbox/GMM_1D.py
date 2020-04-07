import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import norm
np.random.seed(0)


X0 = np.random.normal(loc=3, scale=3, size=50)
X1 = np.random.normal(loc=1, scale=4, size=50)
X2 = np.random.normal(loc=6, scale=5, size=50)
X_tot = np.stack((X0, X1, X2)).flatten()  # Combine the clusters to get the random datapoints from above


class GM1D:

    def __init__(self, X, iterations):
        self.iterations = iterations
        self.X = X
        self.mu = None
        self.pi = None
        self.var = None

    def run(self):
        """
        Instantiate the random mu, pi and var
        """
        self.mu = [-1, 8, 5]
        self.pi = [1 / 3, 1 / 3, 1 / 3]
        self.var = [8, 6, 7]

        """
        E-Step
        """

        for iter in range(self.iterations):

            """Create the array r with dimensionality nxK"""
            r = np.zeros((len(X_tot), 3))

            """
            Probability for each datapoint x_i to belong to gaussian g 
            """
            for c, g, p in zip(range(3), [norm(loc=self.mu[0], scale=self.var[0]),
                                          norm(loc=self.mu[1], scale=self.var[1]),
                                          norm(loc=self.mu[2], scale=self.var[2])], self.pi):
                r[:, c] = p * g.pdf(X_tot)  # Write the probability that x belongs to gaussian c in column c.
                # Therewith we get a 60x3 array filled with the probability that each x_i belongs to one of the gaussians
            """
            Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to 
            cluster c
            """
            for i in range(len(r)):
                r[i] = r[i] / (np.sum(self.pi) * np.sum(r, axis=1)[i])

            """Plot the data"""

            fig = plt.figure(figsize=(5, 5))
            ax0 = fig.add_subplot(111)

            for i in range(len(r)):
                ax0.scatter(self.X[i], 0, c=np.array([r[i][0], r[i][1], r[i][2]]), s=100)

            """Plot the gaussians"""

            for g, c in zip([norm(loc=self.mu[0], scale=self.var[0]).pdf(np.linspace(-20, 20, num=150)),
                             norm(loc=self.mu[1], scale=self.var[1]).pdf(np.linspace(-20, 20, num=150)),
                             norm(loc=self.mu[2], scale=self.var[2]).pdf(np.linspace(-20, 20, num=150))], ['r', 'g', 'b']):
                ax0.plot(np.linspace(-20, 20, num=150), g, c=c)

                print("self.mu", self.mu[0])
                print("self.mu", self.mu[1])
                print("self.mu", self.mu[2])
                print("self.var", self.var[0])
                print("self.var", self.var[1])
                print("self.var", self.var[2])

            """M-Step"""

            """calculate m_c"""
            m_c = []
            for c in range(len(r[0])):
                m = np.sum(r[:, c])
                m_c.append(m)  # For each cluster c, calculate the m_c and add it to the list m_c

            """calculate pi_c"""
            for k in range(len(m_c)):
                self.pi[k] = (m_c[k] / np.sum(m_c))  # For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
                print("shape", np.shape(self.pi))

            """calculate mu_c"""
            self.mu = np.sum(self.X.reshape(len(self.X), 1) * r, axis=0) / m_c

            """calculate var_c"""
            var_c_fake = []

            for c in range(len(r[0])):
                var_c_fake.append((1 / m_c[c]) * np.dot(((np.array(r[:, c]).reshape(150, 1)) * (self.X.reshape(len(self.X), 1) - self.mu[c])).T, (self.X.reshape(len(self.X), 1) - self.mu[c])))

            plt.show()

            var_c = []

            var_c.append(var_c_fake[0][0][0])
            var_c.append(var_c_fake[1][0][0])
            var_c.append(var_c_fake[2][0][0])
            self.var = var_c
            print("self.var", self.var)

        print("dekh",self.mu)
        print("dekh2",self.var)


GM1D = GM1D(X_tot, 5)
GM1D.run()


