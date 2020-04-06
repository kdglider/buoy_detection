'''
Copyright (c) 2020 Hao Da (Kevin) Dong, Anshuman Singh
@file       GMM.py
@date       2020/04/02
@brief      Gaussian Mixture Model class with methods to train and predict
@license    This project is released under the BSD-3-Clause license.
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


'''
@brief      Standard Gaussian Mixture Model with methods to train and predict
'''
class GMM:
    iterations = 0                  # Max number of iterations to train for
    numClusters = 0                 # Number of clusters for GMM to have (N)
    trainingSet = np.array([])      # Training set (MxD NumPy array)
    muArray = np.array([])          # Array of cluster mean vectors (NxD)
    covArray = np.array([])         # Array of cluster covariance matrices (NxDxD)
    piArray = np.array([])          # Array of cluster weights (Pi values) (Nx1)
    gaussianArray = np.array([])    # Array of multivariate Gaussian components of the GMM


    def __init__(self):
        pass
        

    '''
    @brief      Trains GMM on a training data set
    @param      trainingSet     Training set
    @param      numClusters     Number of clusters
    @param      iterations      Max iterations
    @param      epsilon         Termination criteria
    @return     None
    '''
    def train(self, trainingSet, numClusters, iterations=100, epsilon=0.005):
        self.trainingSet = trainingSet
        self.numClusters = numClusters
        self.iterations = iterations

        # Initialize random mu's
        self.muArray = np.random.randint(min(self.trainingSet[:,0]), \
                                         max(self.trainingSet[:,0]), \
                                         size=(self.numClusters, self.trainingSet.shape[1]))

        # Initialize covariance matrices as diagonal
        self.covArray = np.zeros((self.numClusters, self.trainingSet.shape[1], self.trainingSet.shape[1]))

        # All diagonal elements are initially set to the variance of one of the channels of the training set
        randomVariance = np.var(self.trainingSet[:,0])
        for i in range(self.numClusters):
            self.covArray[i] = randomVariance * np.identity(self.trainingSet.shape[1])
        
        # Initialize equal normalized weights
        self.piArray = np.ones(self.numClusters) / self.numClusters

        # Store the log likehoods per iteration and plot in the end to check convergence
        log_likelihood_errors = np.zeros(self.iterations)
        
        # Responsibility values matrix (probabilities that each sample belongs to a cluster)
        R = np.zeros((self.trainingSet.shape[0], self.numClusters))

        # Train GMM for max iterations or until the log-likelihood error change is < epsilon
        iteration = 0
        errorChange = float('inf')
        while(iteration < iterations and errorChange > epsilon):

            ## Expectation Step ##
            # Update all responsibility values in R matrix
            for mu, cov, pi, c in zip(self.muArray, self.covArray, self.piArray, range(self.numClusters)):
                mn = multivariate_normal(mean=mu, cov=cov)
                R[:, c] = pi * mn.pdf(self.trainingSet) /   \
                    np.sum([pi_c * multivariate_normal(mean=mu_c, cov=cov_c).pdf(self.trainingSet) \
                        for pi_c, mu_c, cov_c in zip(self.piArray, self.muArray, self.covArray)], axis=0)

            ## Maximization Step ##
            # Update the mean vectors, covariance matrices and weights based on the 
            # responsibility values

            for c in range(self.numClusters):
                # Total responsibility of cluster c
                m_c = np.sum(R[:, c])

                # Update pi_c = m_c / m
                self.piArray[c] = m_c / self.trainingSet.shape[0]

                # Update mu_c = 1/m_c * sum_i(r_ic * x_i)
                mu_c = np.zeros(self.trainingSet.shape[1])
                for r_ic, x_i in zip(R[:, c], self.trainingSet):
                    mu_c += (1 / m_c) * r_ic * x_i

                self.muArray[c] = mu_c

            # Update covariance matrices in a separate loop since the calculations depend on 
            # the new weighted means
            for c in range(self.numClusters):
                # Total responsibility of cluster c
                m_c = np.sum(R[:, c])
                
                # Reshape mean into column vector
                mu_c = np.reshape(self.muArray[c], (self.muArray[c].shape[0],1))

                # Update cov_c = 1/m_c * sum_i(r_ic * (x_i - mu_c) * (x_i - mu_c)^T)
                cov_c = np.zeros((self.trainingSet.shape[1], self.trainingSet.shape[1]))
                for r_ic, x_i in zip(R[:, c], self.trainingSet):
                    # Reshape x_i into column vector
                    x_i = np.reshape(x_i, (x_i.shape[0],1))
                    cov_c += (r_ic / m_c) * np.matmul((x_i - mu_c), (x_i - mu_c).T)

                self.covArray[c] = cov_c

            # Record the log-likelihood error
            log_likelihood_errors[iteration] = -np.log(np.sum([self.piArray[c] * \
                multivariate_normal(self.muArray[c], self.covArray[c]).pdf(self.trainingSet) \
                    for c in range(self.numClusters)]))

            # Calculate change in error if the iteration is greater than 0
            if (iteration > 0):
                errorChange = log_likelihood_errors[iteration-1] - log_likelihood_errors[iteration]
            
            iteration += 1
        
        # Trim any trailing zeros from log_likelihood_erros
        log_likelihood_errors = np.trim_zeros(log_likelihood_errors)

        # Record GMM component Gaussians
        gaussianList = []
        for c in range(self.numClusters):
            mn = multivariate_normal(self.muArray[c], self.covArray[c])
            gaussianList.append(mn)
        self.gaussianArray = np.array(gaussianList)
    
        # Plot log-likelihood error against iterations
        plt.plot(np.arange(0, log_likelihood_errors.shape[0], 1), log_likelihood_errors)
        plt.title('Log-Likelihood Error')
        plt.xlabel('Iterations')
        plt.ylabel('-log(Probability)')
        plt.show()
        

    '''
    @brief      Trains GMM on a training data set
    @param      sample      NumPy array of sample data
    @return     result      NumPy array of log-likelihood errors of sample data
    '''
    def getLogLikelihoodError(self, sample):
        # Return 0 if the GMM has not been trained
        if (self.numClusters == 0):
            print('GMM has not been trained')
            return 0

        # Compute total log-likelihood error with GMM
        result = -np.log(np.sum([self.piArray[c] * self.gaussianArray[c].pdf(sample) \
            for c in range(self.numClusters)], axis=0))

        return result


    '''
    @brief      Saves current GMM parameters to a file
    @param      filename    Name of parameter file
    @return     None
    '''
    def save(self, filename):
        np.savez(filename, \
            iterations = self.iterations, \
            numClusters = self.numClusters, \
            muArray = self.muArray, \
            covArray = self.covArray, \
            piArray = self.piArray, \
            gaussianArray = self.gaussianArray)

    
    '''
    @brief      Loads GMM parameters from a saved file
    @param      filename    Name of parameter file
    @return     None
    '''
    def load(self, filename):
        params = np.load(filename, allow_pickle=True)

        self.iterations = params['iterations']
        self.numClusters = params['numClusters']
        self.muArray = params['muArray']
        self.covArray = params['covArray']
        self.piArray = params['piArray']
        self.gaussianArray = params['gaussianArray']


if __name__ == '__main__':
    # Load a training set
    trainingSet = np.load('training_set/yellowTrainingSet.npy')

    gmm = GMM()

    # Train GMM
    gmm.train(trainingSet, numClusters=2, iterations=10)
    
    # Save parameters
    gmm.save('gmmParams.npz')

    
    
