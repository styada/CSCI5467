import numpy as np

class Normalization:
    def __init__(self,):
        self.mean = np.zeros([1,64]) # means of training features
        self.std = np.zeros([1,64]) # standard deviation of training features

    def fit(self, x):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)
        # compute the statistics of training samples (i.e., means and std)

    def normalize(self, x):
        x = (x - self.mean) / (1e-15 + self.std)
        # normalize the given samples to have zero mean and unit variance (add 1e-15 to std to avoid numeric issue)
        return x

def process_label(label):
    # convert the labels into one-hot vector for training
    one_hot = np.zeros([len(label), 10])
    one_hot[np.arange(len(label)), label] = 1 # we have our array label define each column and index accordingly

    return one_hot

def tanh(x):
    # implement the hyperbolic tangent activation function for hidden layer
    x = np.clip(x,a_min=-100,a_max=100) # for stablility, do not remove this line

    f_x = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return f_x

def softmax(x):
    # implement the softmax activation function for output layer
    e_x = np.exp(x - np.max(x)) # no overflow
    f_x = e_x / np.sum(e_x, axis=1).reshape(-1, 1)
    return f_x

class MLP:
    def __init__(self,num_hid):
        # initialize the weights
        self.weight_1 = np.random.random([64,num_hid])
        self.bias_1 = np.random.random([1,num_hid])
        self.weight_2 = np.random.random([num_hid,10])
        self.bias_2 = np.random.random([1,10])

    def fit(self,train_x,train_y, valid_x, valid_y):
        # learning rate
        lr = 5e-3
        # counter for recording the number of epochs without improvement
        count = 0
        best_valid_acc = 0

        """
        Stop the training if there is no improvement over the best validation accuracy for more than 50 iterations
        """
        while count <= 50:
            # training with all samples (full-batch gradient descents)
            # implement the forward pass (from inputs to predictions)
            hidden_layer = np.dot(train_x, self.weight_1) + self.bias_1
            zh = tanh(hidden_layer)
            output = np.dot(zh, self.weight_2) + self.bias_2
            yt = softmax(output)

            # implement the backward pass (backpropagation)
            # compute the gradients w.r.t. different parameters
            grad_vh = zh.T.dot(train_y - yt)
            grad_v0 = np.sum((train_y - yt), axis=0)
            w = (train_y - yt).dot(self.weight_2.T) * (1 - np.power(zh, 2))
            grad_whj = train_x.T.dot(w)
            grad_wh0 = np.sum(w, axis=0)

            # update the parameters based on sum of gradients for all training samples
            self.weight_1 = self.weight_1 + lr * grad_whj
            self.weight_2 = self.weight_2 + lr * grad_vh
            self.bias_1 = self.bias_1 + lr * grad_wh0
            self.bias_2 = self.bias_2 + lr * grad_v0

            # evaluate on validation data
            predictions = self.predict(valid_x)
            valid_acc = np.count_nonzero(predictions.reshape(-1)==valid_y.reshape(-1))/len(valid_x)

            # compare the current validation accuracy with the best one
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                count = 0
            else:
                count += 1

        return best_valid_acc

    def predict(self, x):
        # generate the predicted probability of different classes
        hidden_layer = np.dot(x, self.weight_1) + self.bias_1
        zh = tanh(hidden_layer)
        output = np.dot(zh, self.weight_2) + self.bias_2
        yt = softmax(output)

        # convert class probability to predicted labels
        y = np.zeros([len(x),]).astype('int') # placeholder
        for i in range(len(y)):
            y[i] = np.argmax(yt[i])

        return y

    def get_hidden(self, x):
        # extract the intermediate features computed at the hidden layers (after applying activation function)
        hidden_layer = np.dot(x, self.weight_1) + self.bias_1
        zh = tanh(hidden_layer)
        z = zh

        return z

    def params(self):
        return self.weight_1, self.bias_1, self.weight_2, self.bias_2
