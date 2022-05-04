#import libraries
import numpy as np

class Kmeans:
    def __init__(self,k=8): # k is number of clusters
        self.num_cluster = k
        self.center = None # centers for different clusters
        self.cluster_label = np.zeros([k,]) # class labels for different clusters
        self.error_history = []

    def fit(self, X, y):        
        # initialize the centers of clutsers as a set of pre-selected samples
        init_idx = [1, 200, 500, 1000, 1001, 1500, 2000, 2005]

        centers = []

        for i in range(len(init_idx)):
            centers.append(X[init_idx[i]])
        self.center = np.array(centers)

        num_iter = 0                                                    # number of iterations for convergence

                                                                        # initialize the cluster assignment
        prev_cluster_assignment = np.zeros([len(X),]).astype('int')
        cluster_assignment = np.zeros([len(X),]).astype('int')
        is_converged = False
        
        diff = np.zeros([len(self.center),])                            # variable created to store distance from mean

                                                                        # iteratively update the centers of clusters till convergence
        while not is_converged:

                                                                        # iterate through the samples and compute their cluster assignment (E step)
            for i in range(len(X)):
                for j in range(len(self.center)):
                                                                        # use euclidean distance to measure the distance between sample and cluster centers
                    diff[j] = np.linalg.norm(X[i] - self.center[j])

                cluster_assignment[i] = np.argmin(diff)                 # determine the cluster assignment by selecting the cluster whose center is closest to the sample

                                                                        # update the centers based on cluster assignment (M step) 
            for i in range(len(self.center)):  
               self.center[i] = np.mean(X[cluster_assignment == i], axis=0)
                                                                        # compute the reconstruction error for the current iteration
            cur_error = self.compute_error(X, cluster_assignment)
            self.error_history.append(cur_error)
                                                                        # reach convergence if the assignment does not change anymore
            is_converged = True if (cluster_assignment==prev_cluster_assignment).sum() == len(X) else False
            prev_cluster_assignment = np.copy(cluster_assignment)
            num_iter += 1
                                                                        # compute the class label of each cluster based on majority voting (remember to update the corresponding class attribute)
            for i in range(self.num_cluster):
                cluster = y[cluster_assignment==i]
                cluster = cluster.tolist()
                self.cluster_label[i] = np.bincount(cluster).argmax()

        return num_iter, self.error_history

    def predict(self,X):
                                                                        # predicting the labels of test samples based on their clustering results
        prediction = np.ones([len(X),]) # placeholder

        diff = np.zeros([len(self.center),])                            # variable created to store distance from mean
                                                                        # iterate through the test samples
        for i in range(len(X)):
                                                                        # find the cluster of each sample
            for j in range(len(self.center)):
                                                                        # use euclidean distance to measure the distance between sample and cluster centers
                    diff[j] = np.linalg.norm(X[i] - self.center[j])
                                                                        # use the class label of the selected cluster as the predicted class
            prediction[i] = self.cluster_label[np.argmin(diff)]

        return prediction

    def compute_error(self,X,cluster_assignment):
                                                                        # compute the reconstruction error for given cluster assignment and centers
        error = 0 # placeholder
        for i in range(len(X)):
            error += np.linalg.norm((X[i]-self.center[cluster_assignment[i]]))**2

        return error

    def params(self):
        return self.center, self.cluster_label

#Questions:
#Lines 79-81
#Line 71
#Lines 15-16  