

import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import multivariate_normal




class GMM:

    def __init__(self, X, number_of_sources, iterations):
        self.iterations = iterations
        self.number_of_sources = number_of_sources
        self.X = X
        self.mu = None
        self.pi = None
        self.covar = None
        self.XY = None

    """Define a function which runs for iterations, iterations"""

    def run(self):
        self.reg_cov = 1e-6 * np.identity(len(self.X[0]))


        """T is used to transpose """

        """ 1. Set the initial mu, covariance and pi values"""
        self.mu = np.random.randint(min(self.X[:, 0]), max(self.X[:, 0]), size=(self.number_of_sources, len(self.X[0])))
        # This is a nxm matrix since we assume n sources (n Gaussians) where each has m dimensions
        """ Return a random integer N such that a <= N <= b"""

        self.covar = np.zeros((self.number_of_sources, len(X[0]), len(X[0])))
        """We need a nxmxm covariance matrix for each source since we have m features
        --> We create symmetric covariance matrices with ones on the digonal"""

        for dim in range(len(self.covar)):
            np.fill_diagonal(self.covar[dim], 5)

            """making diagonal of each dimension (channel) of self.covar, equal to 5"""

        self.pi = np.ones(self.number_of_sources) / self.number_of_sources  # Are "Fractions"
        print("self.pi", np.shape(self.pi))
        log_likelihoods = []
        # In this list we store the log likehoods per iteration and plot them in the end to check if we have converged

        for m, c in zip(self.mu, self.covar):
            c += self.reg_cov
            multi_normal = multivariate_normal(mean=m, cov=c)
            # Draw random samples from a multivariate normal distribution

        for i in range(self.iterations):

            """E Step"""
            r_ic = np.zeros((len(self.X), len(self.covar)))

            for m, co, p, r in zip(self.mu, self.covar, self.pi, range(len(r_ic[0]))):
                co += self.reg_cov
                mn = multivariate_normal(mean=m, cov=co)
                print("mn", np.shape(p * mn.pdf(self.X)))
                r_ic[:, r] = p * mn.pdf(self.X) / np.sum([pi_c * multivariate_normal(mean=mu_c, cov=cov_c).pdf(X) for pi_c, mu_c, cov_c in zip(self.pi, self.mu, self.covar + self.reg_cov)], axis=0)

            print("xxxx", np.shape(r_ic))
            """norm.pdf returns a Probability Density Function value from a normal distribution for each value of argument of pdf()"""
            """multivariate_normal.pdf returns a Probability Density Function value from a normal multivariate distribution for each value of self.X"""
            """ using one liner for loop to create list"""
            """Example : squares = [i**2 for i in range(10)]"""
            """Example : squares = [i**2 for i in range(10)]"""

            """M Step"""

            # Calculate the new mean vector and new covariance matrices, based on the probable membership of the single x_i to classes c --> r_ic
            self.mu = []
            self.covar = []
            self.pi = []
            log_likelihood = []

            for c in range(len(r_ic[0])):
                m_c = np.sum(r_ic[:, c], axis=0)
                mu_c = (1 / m_c) * np.sum(self.X * r_ic[:, c].reshape(len(self.X), 1), axis=0)
                self.mu.append(mu_c)

                # Calculate the covariance matrix per source based on the new mean
                self.covar.append(((1 / m_c) * np.dot((np.array(r_ic[:, c]).reshape(len(self.X), 1) * (self.X - mu_c)).T, (self.X - mu_c))) + self.reg_cov)
                # Calculate pi_new which is the "fraction of points" respectively the fraction of the probability assigned to each source
                self.pi.append(m_c / np.sum(r_ic))  # Here np.sum(r_ic) gives as result the number of instances. This is logical since we know
                # that the columns of each row of r_ic adds up to 1. Since we add up all elements, we sum up all
                # columns per row which gives 1 and then all rows which gives then the number of instances (rows)
                # in X --> Since pi_new contains the fractions of datapoints, assigned to the sources c,
                # The elements in pi_new must add up to 1

            """Log likelihood"""
            log_likelihoods.append(np.log(np.sum([k * multivariate_normal(self.mu[i], self.covar[j]).pdf(X) for k, i, j in zip(self.pi, range(len(self.mu)), range(len(self.covar)))])))

            """
            This process of E step followed by a M step is now iterated a number of n times. In the second step for instance,
            we use the calculated pi_new, mu_new and cov_new to calculate the new r_ic which are then used in the second M step
            to calculat the mu_new2 and cov_new2 and so on....
            """

        fig2 = plt.figure(figsize=(10, 10))
        ax1 = fig2.add_subplot(111)
        ax1.set_title('Log-Likelihood')
        ax1.plot(range(0, self.iterations, 1), log_likelihoods)
        plt.show()

    """Predict the membership of an unseen, new datapoint"""

    def predict(self, Y):

        for m, c in zip(self.mu, self.covar):
            multi_normal = multivariate_normal(mean=m, cov=c)

        prediction = []
        for m, c in zip(self.mu, self.covar):
            # print(c)
            prediction.append(multivariate_normal(mean=m, cov=c).pdf(Y) / np.sum([multivariate_normal(mean=mean, cov=covar).pdf(Y) for mean, covar in zip(self.mu, self.covar)]))
        print("prediction", prediction)
        return prediction


if __name__ == '__main__':
    # Load a training set
    trainingSet = np.load('training_set/yellowTrainingSet.npy')
    print(trainingSet)
    print(trainingSet.tolist())

    GMM = GMM(trainingSet.tolist(), 2, 50)
    GMM.run()
    GMM.predict([[100, 100, 100]])
