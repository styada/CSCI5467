import numpy as np

class PCA():
    def __init__(self,num_dim=None):
        self.num_dim = num_dim
        self.mean = np.zeros([1,784]) # means of training data
        self.W = None # projection matrix

    def fit(self,X):
        # normalize the data to make it centered at zero (also store the means as class attribute)
        #Find mean store in self.mean
        self.mean = np.mean(X,axis = 0)

        for i in range(len(X)):
            X[i] = X[i]- self.mean

        projection_matrix = (1/(len(X)-1)) * X.T@X # finding the projection matrix that maximize the variance (Hint: for eigen computation, use numpy.eigh instead of numpy.eig)

        vals, vecs = np.linalg.eigh(projection_matrix)   # store the projected dimension

        vals = vals[::-1]

        if self.num_dim is None: 
            vals_reduced = []
            num_dim = 0
            i = 0
            while((np.sum(vals_reduced)/np.sum(vals))<0.9):
                vals_reduced.append(vals[i])
                num_dim += 1 # select the reduced dimension that keep >90% of the variance
                i += 1

            self.num_dim = num_dim

        self.W = vecs[:,::-1]# determine the projection matrix and store it as class attribute
        self.W = self.W[:,:self.num_dim]

        X_pca = X @ self.W # W * x  # project the high-dimensional data to low-dimensional one

        return X_pca, self.num_dim

    def predict(self,X):
        # normalize the test data based on training statistics
        for i in range(len(X)):
            X[i] = X[i]- self.mean

        W = self.W[:self.num_dim]
        
        X_pca = X @ self.W  # W^T * x # project the test data 
                           
        return X_pca

    def params(self):
        return self.W, self.mean, self.num_dim


