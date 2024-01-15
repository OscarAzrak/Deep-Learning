#!/usr/bin/env python
# coding: utf-8

# In[117]:


import numpy as np
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
 
 #From updated functions.py from Canvas. Changed names on functions from previous version.
 
def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def LoadBatch(filename):
    """ Copied from the dataset website """
    with open('cifar-10-batches-py/' + filename, 'rb') as fo:
        dataset_dict = pickle.load(fo, encoding='bytes')
 
    return dataset_dict
 
def montage(W):
    """ Display the image for each label in W """
    fig, ax = plt.subplots(2, 5)
    for i in range(2):
        for j in range(5):
            im = W[i * 5 + j, :].reshape(32, 32, 3, order='F')
            sim = (im - np.min(im[:])) / (np.max(im[:]) - np.min(im[:]))
            sim = sim.transpose(1, 0, 2)
            ax[i][j].imshow(sim, interpolation='nearest')
            ax[i][j].set_title("y=" + str(5 * i + j))
            ax[i][j].axis('off')
    plt.show()
 
 
def save_as_mat(data, name="model"):
    """ Used to transfer a python model to matlab """
    import scipy.io as sio
    sio.savemat(f'{name}.mat', {"name": "b"})


def load(filename):
    data = LoadBatch(filename)
    
    X = data[b'data'].astype(np.float32).reshape(-1, 3072).T / 255
    y = np.array(data[b'labels']).astype(np.int32)
    
    Y = np.zeros((10, y.shape[0]), dtype=np.float32)
    for i, label in enumerate(y):
        Y[label, i] = 1
    
    return X, Y, y + 1


def initialize_weights(shapes_list):
    np.random.seed(369)

    weights = []
    biases = []

    for shape in shapes_list:
        W = np.random.normal(0, 0.001, size=(shape[0], shape[1]))
        b = np.zeros(shape=(shape[0], 1))

        weights.append(W)
        biases.append(b)

    return weights, biases



def normalize_data(train, validation, test):
    # Compute the mean and standard deviation of the training data
    train_mean = np.mean(train, axis=1, keepdims=True)
    train_std = np.std(train, axis=1, keepdims=True)

    # Normalize the training, validation and test data
    train_norm = (train - train_mean) / train_std
    validation_norm = (validation - train_mean) / train_std
    test_norm = (test - train_mean) / train_std

    return train_norm, validation_norm, test_norm


# In[118]:


def he_initialization_k_layers(shapes_list):
    weights = []
    biases = []

    for pair in shapes_list:

        weights.append(np.random.randn(pair[0], pair[1]) * np.sqrt(2 / float(pair[0])))
        biases.append(np.zeros(shape=(pair[0], 1)))

    return weights, biases


# In[119]:


def compute_loss(Y, P, num_samples):
    return -np.sum(Y * np.log(P)) / num_samples

def compute_regularization(lamda, W):
    return sum([lamda * np.sum(W_l ** 2) for W_l in W])

def ComputeCost(X, Y, W, b, lamda):

    P, _ = forward_pass(X, W, b)

    loss = compute_loss(Y, P, X.shape[1])
    reg = compute_regularization(lamda, W)
    cost = loss + reg

    return cost, loss


def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(S):
    H = S.copy()
    H[H<0] = 0
    return H

def forward_pass(X, W, b):
    num_layers = len(W)
    X_layer =[X.copy()]+[None]*(num_layers-1)
    S = [None]*(num_layers-1)

    for layer in range(num_layers-1):
        S[layer] = np.dot(W[layer], X_layer[layer]) + b[layer]
        X_layer[layer+1] = relu(S[layer])
            
    P = softmax(np.matmul(W[num_layers-1], X_layer[num_layers-1]) + b[num_layers-1])

    return P, X_layer


# In[120]:


def ComputeAccuracy(X, y, W, b):
    P, _ = forward_pass(X, W, b)
    predictions = np.argmax(P, axis=0)
    accuracy = np.sum(predictions == y) / X.shape[1]
    return accuracy


# In[121]:


def ComputeGradients(X, Y, W, b, lambda_):
    num_layers = len(W)
    
    # initialize gradients
    grad_W = [None] * num_layers
    grad_b = [None] * num_layers
    P, X_layer = forward_pass(X, W, b)
    G = P - Y

    grad_W[num_layers-1] = np.matmul(G, X_layer[num_layers-1].T) / X.shape[1] + 2 * lambda_ * W[num_layers-1]
    grad_b[num_layers-1] = np.matmul(G, np.ones((X.shape[1], 1))) / X.shape[1]
    for layer in range(num_layers-2, -1, -1):
        G = np.matmul(W[layer+1].T, G) * (X_layer[layer+1] > 0)
        grad_W[layer] = np.matmul(G, X_layer[layer].T) / X.shape[1] + 2 * lambda_ * W[layer]
        grad_b[layer] = np.matmul(G, np.ones((X.shape[1], 1))) / X.shape[1]

    return grad_W, grad_b


# In[122]:


def compute_gradients_num(X,Y, W, b, lambda_, h):
    grad_weights = []
    grad_bias = []

    for layer in tqdm(range(len(W))):
        weights = W[layer]
        bias = b[layer]

        grad_W = np.zeros(weights.shape)
        grad_b = np.zeros(bias.shape)


        #W
        for i in (range(0, weights.shape[0])):
            for j in range(weights.shape[1]):
                W_ = np.copy(weights)
                W_[i,j] -= h
                temp_weights= W.copy()
                temp_weights[layer] = W_
                c1, _ = ComputeCost(X, Y, temp_weights, b, lambda_)
                W_ = np.copy(weights)
                W_[i,j] += h
                temp_weights= W.copy()
                temp_weights[layer] = W_
                c2, _= ComputeCost(X, Y, temp_weights, b, lambda_)
                grad_W[i,j] = (c2 - c1) / (2*h)
        
        grad_weights.append(grad_W)

        for i in tqdm(range(bias.shape[0])):
            b_ = np.copy(bias)
            b_[i, 0] -= h
            temp_bias = b.copy()
            temp_bias[layer] = b_
            c1, _ = ComputeCost(X, Y, W, temp_bias, lambda_)
                
            b_ = np.copy(bias)
            b_[i, 0] += h
            temp_bias = b.copy()
            temp_bias[layer] = b_
            c2 = ComputeCost(X, Y, W, temp_bias, lambda_)[0]
            grad_b[i, 0] = (c2 - c1) / (2*h)

        grad_bias.append(grad_b)

    return grad_weights, grad_bias


# In[123]:


def ComputeTestingGradient(grad_weights, grad_biases, num_weights, num_biases):
    for layer in range(len(grad_weights)):
        grad_w = grad_weights[layer]
        grad_b = grad_biases[layer]
        num_w = num_weights[layer]
        num_b = num_biases[layer]
        print(f'Layer {layer}')
        print(f'Weight difference: {np.max(np.abs(grad_w - num_w))}')
        print(f'Bias difference: {np.max(np.abs(grad_b - num_b))}')


# In[8]:


#testing for 2 layers

W, b = initialize_weights([(50, 3072), (10, 50)])


X_train, Y_train, y_train = load('data_batch_1')
X_validation, Y_validation, y_validation = load('data_batch_2')
X_test, Y_test, y_test = load('test_batch')

X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)

lambda_ = 0

grad_weights, grad_biases = ComputeGradients(X_train[:, :100], Y_train[:, :100], W, b, lambda_)

grad_weights_num, grad_biases_num = compute_gradients_num(X_train[:, :100], Y_train[:, :100], W, b, lambda_, 1e-5)

ComputeTestingGradient(grad_weights, grad_biases, grad_weights_num, grad_biases_num)


# In[9]:


#ComputeTestingGradient(grad_weights, grad_biases, grad_weights_num, grad_biases_num)


# In[10]:


# initalize for layer 3
W, b = initialize_weights([(50, 3072), (20, 50), (10, 20)])

grad_weights, grad_biases = ComputeGradients(X_train[:, :100], Y_train[:, :100], W, b, lambda_)
grad_weights_num, grad_biases_num = compute_gradients_num(X_train[:, :100], Y_train[:, :100], W, b, lambda_, 1e-5)

ComputeTestingGradient(grad_weights, grad_biases, grad_weights_num, grad_biases_num)


# In[11]:


# initalise for layer 4

W, b = initialize_weights([(50, 3072), (20, 50), (15, 20), (10, 15)])

grad_weights, grad_biases = ComputeGradients(X_train[:, :100], Y_train[:, :100], W, b, lambda_)
grad_weights_num, grad_biases_num = compute_gradients_num(X_train[:, :100], Y_train[:, :100], W, b, lambda_, 1e-5)

ComputeTestingGradient(grad_weights, grad_biases, grad_weights_num, grad_biases_num)


# In[124]:


class CyclicalLearningRate:
    def __init__(self, min_lr, max_lr, total_epochs, step_size):
        import math

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_epochs = total_epochs
        self.step_multiplier = 2
        self.cycle_multiplier = 3
        self.step_duration = self.step_multiplier * math.floor(step_size)
        self.time = 2 * self.cycle_multiplier * self.step_duration
        self.current_cycle = 1

    def __call__(self):
        self.time += 1

        # Check if we are in a new cycle
        if self.time > 2 * (self.cycle_multiplier + 1) * self.step_duration:
            self.cycle_multiplier += 1
            self.current_cycle += 1

        if (
            self.time >= 2 * self.cycle_multiplier * self.step_duration
        ) and (
            self.time <= (2 * self.cycle_multiplier + 1) * self.step_duration
        ):
            lr = self.min_lr + 1 / self.step_duration * (
                self.time - 2 * self.cycle_multiplier * self.step_duration
            ) * (self.max_lr - self.min_lr)
            return lr

        elif (
            self.time >= (2 * self.cycle_multiplier + 1) * self.step_duration
        ) and (
            self.time <= 2 * (self.cycle_multiplier + 1) * self.step_duration
        ):
            lr = self.max_lr - 1 / self.step_duration * (
                self.time - (2 * self.cycle_multiplier + 1) * self.step_duration
            ) * (self.max_lr - self.min_lr)
            return lr

        else:
            raise ValueError("Issue in cyclical learning rate calculation")



# In[125]:


def randomize_data_order(data_x, data_y, labels):
    import numpy as np

    idx = np.arange(data_x.shape[1])
    np.random.shuffle(idx)
    return data_x[:, idx], data_y[:, idx], labels[idx]

def train_model(W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size):
    training_cost = []
    training_loss = []
    training_accuracy = []
    validation_cost = []
    validation_accuracy = []
    validation_loss = []
    X_train, Y_train, y_train = dataset["train"]
    X_val, Y_val, y_val = dataset["val"]
    num_steps = X_train.shape[1] / batch_size
    #print(X_train.shape[1])
    #num_steps = (5*45000) / batch_size


    clr = CyclicalLearningRate(min_lr, max_lr, n_epochs, num_steps)
    lr = min_lr

    while clr.current_cycle < 3:
        # Randomize the data
        X_train, Y_train, y_train = randomize_data_order(X_train, Y_train, y_train)

        #for batch_idx in tqdm(range(int(n_epochs))):
        for batch_idx in tqdm(range(int(num_steps))):


            
            X_batch = X_train[:, batch_idx * batch_size : (batch_idx + 1) * batch_size]
            Y_batch = Y_train[:, batch_idx * batch_size : (batch_idx + 1) * batch_size]

            grad_W, grad_b = ComputeGradients(X_batch, Y_batch, W, b, lamda)
            # Update weights and biases
            W = [W[layer] - lr*grad_W[layer] for layer in range(len(W))]
            b = [b[layer] - lr*grad_b[layer] for layer in range(len(b))]

            lr = clr()

        training_cost.append(ComputeCost(X_train, Y_train, W, b, lamda)[0])
        training_loss.append(ComputeCost(X_train, Y_train, W, b, lamda)[1])
        training_accuracy.append(ComputeAccuracy(X_train, y_train, W, b))
        validation_cost.append(ComputeCost(X_val, Y_val, W, b, lamda)[0])
        validation_loss.append(ComputeCost(X_val, Y_val, W, b, lamda)[1])
        validation_accuracy.append(ComputeAccuracy(X_val, y_val, W, b))

    return W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost



# In[126]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)



def load_data(file_name):
    """Adapted from the dataset website"""
    import pickle

    file_path = "cifar-10-batches-py/" + file_name

    with open(file_path, "rb") as file_obj:
        data_dict = pickle.load(file_obj, encoding="bytes")
    return data_dict

def load_dataset():
    data_partitions = {
        "train": "data_batch_1",
        "test": "test_batch",
        "val": "data_batch_2",
    }

    for key, file in data_partitions.items():
        data = load_data(file)

        num_samples = len(data[b"data"])
        X = np.array(data[b"data"]).T
        Y = encoder.fit_transform(np.array(data[b"labels"]).reshape(-1, 1)).T
        y = np.array(data[b"labels"])

        data_partitions[key] = [X, Y, y]

    preprocess_data(data_partitions)

    return data_partitions

def preprocess_data(data_partitions):
    mean_X = np.mean(data_partitions["train"][0], axis=1).reshape(-1, 1)
    std_X = np.std(data_partitions["train"][0], axis=1).reshape(-1, 1)

    for key in data_partitions.keys():
        data_partitions[key][0] = (data_partitions[key][0] - mean_X) / std_X

    return data_partitions

def load_full_dataset():
    data_partitions = {
        "train": ["data_batch_1", "data_batch_3", "data_batch_4", "data_batch_5"],
        "test": "test_batch",
        "val": "data_batch_2",
    }

    for key, file in data_partitions.items():
        X, Y, y = [], [], []
        if isinstance(file, list):
            for fi in file:
                data = load_data(fi)
                num_samples = len(data[b"data"])
                X.append(np.array(data[b"data"]).T)
                Y.append(encoder.fit_transform(np.array(data[b"labels"]).reshape(-1, 1)).T)
                y.append(np.array(data[b"labels"]))
            X = np.concatenate(X, axis=1)
            Y = np.concatenate(Y, axis=1)
            y = np.concatenate(y, axis=0)

        else:
            data = load_data(file)
            num_samples = len(data[b"data"])
            X = np.array(data[b"data"]).T
            Y = encoder.fit_transform(np.array(data[b"labels"]).reshape(-1, 1)).T
            y = np.array(data[b"labels"])

        data_partitions[key] = [X, Y, y]

    preprocess_data(data_partitions)

    return data_partitions


# In[15]:


# Exercise 2: replicate default resuts form assignment 2
W, b = he_initialization_k_layers([(50, 3072), (10, 50)])

dataset = load_dataset()

lamda = 0.004641588833612777 # best lambda from previous assignment
min_lr = 1e-5
max_lr = 1e-1
batch_size = 100
n_epochs = 200

W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost =  train_model(
    W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)


# In[16]:


# accuracy
plt.plot(training_accuracy, label='Training accuracy')
plt.plot(validation_accuracy, label='Test accuracy')
plt.legend()
plt.show()


# In[45]:


# Exercise 2: 3-layer network with 50 and 50 nodes
W, b = he_initialization_k_layers([[50, 3072], [50, 50], [10, 50]])
min_lr = 1e-5
max_lr = 1e-1
lamda = 0.005
batch_size = 100
dataset = load_dataset()
X_train, Y_train, y_train = dataset["train"]
X_val, Y_val, y_val = dataset["val"]
n_epochs = 2 * int(5 * X_train.shape[1] / batch_size)  # Two cycles



W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost =  train_model(
    W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)


# In[46]:


# accuracy
plt.plot(training_accuracy, label='Training accuracy')
plt.plot(validation_accuracy, label='Test accuracy')
plt.legend()
plt.show()


# In[103]:


plt.plot(training_cost, label='Training cost')
plt.plot(validation_cost, label='Validation cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.savefig('cost.png')
plt.show()

plt.plot(training_loss, label='Training loss')
plt.plot(validation_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
plt.show()


# In[52]:


# Exercise 2: 9-layer network
shapes_list = [(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)]
W, b = he_initialization_k_layers(shapes_list)

min_lr = 1e-5
max_lr = 1e-1
lamda = 0.005
batch_size = 100
dataset = load_dataset()
X_train, Y_train, y_train = dataset["train"]
X_val, Y_val, y_val = dataset["val"]
n_epochs = 2 * int(5 * X_train.shape[1] / batch_size)  # Two cycles



# In[53]:


W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost =  train_model(
    W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)


# In[54]:


plt.plot(training_accuracy, label='Training accuracy')
plt.plot(validation_accuracy, label='Test accuracy')
plt.legend()
plt.show()
print(validation_accuracy[-1])


# In[55]:


# exercise 3


# In[40]:


def relu2(x):


    return np.maximum(x, 0)


# In[41]:


def BatchNormalize(s, mean_s, var_s, epsilon=1e-20):

    diff = s - mean_s

    return diff / (np.sqrt(var_s + epsilon))


# In[50]:


def ForwardPassBatchNormalization(X, weights, biases, exponentials= None):

    s = np.dot(weights[0], X) + biases[0]

    intermediate_outputs = [s]

    if exponentials is not None:

        exponential_means = exponentials[0]
        exponential_variances = exponentials[1]

        mean_s = exponential_means[0]
        var_s = exponential_variances[0]

    else:

        mean_s = s.mean(axis=1).reshape(s.shape[0], 1)
        var_s = s.var(axis=1).reshape(s.shape[0], 1)

        means = [mean_s]
        variances = [var_s]

    normalized_score = BatchNormalize(s, mean_s, var_s)

    batch_normalization_outputs = [normalized_score]
    batch_normalization_activations = [relu2(normalized_score)]

    for index in range(1, len(weights) - 1):

        s = np.dot(weights[index], batch_normalization_activations[-1]) + biases[index]

        intermediate_outputs.append(s)

        if exponentials is None:
            mean_s = s.mean(axis=1).reshape(s.shape[0], 1)
            var_s = s.var(axis=1).reshape(s.shape[0], 1)

            means.append(mean_s)
            variances.append(var_s)

        else:

            mean_s = exponential_means[index]
            var_s = exponential_variances[index]

        normalized_score = BatchNormalize(s, mean_s, var_s)

        batch_normalization_outputs.append(normalized_score)
        batch_normalization_activations.append(relu2(normalized_score))

    s = np.dot(weights[-1], batch_normalization_activations[-1]) + biases[-1]

    p = softmax(s)

    if exponentials is not None:
        return p
    else:
        return p, batch_normalization_activations, batch_normalization_outputs, intermediate_outputs, means, variances


# In[88]:


def ComputeAccuracyBatchNormalization(X, y, weights, biases, exponentials = None):

    if exponentials is not None:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)
    else:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)[0]
    predictions = np.argmax(p, axis=0)

    accuracy = round(np.sum(np.where(predictions - y == 0, 1, 0)) * 100 / len(y), 2)

    return accuracy


# In[80]:


def ComputeCostBatchNormalization(X, Y, weights, biases, regularization_term, exponentials=None):


    if exponentials is not None:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)
    else:
        p = ForwardPassBatchNormalization(X, weights, biases, exponentials)[0]

    cross_entropy_loss = -np.log(np.diag(np.dot(Y.T, p))).sum() / float(X.shape[1])

    weight_sum = 0
    for weight in weights:

        weight_sum += np.power(weight, 2).sum()

    return cross_entropy_loss + regularization_term * weight_sum, cross_entropy_loss 


# In[81]:


def BatchNormBackPass(g, s, mean_s, var_s, epsilon=1e-20):

    # First part of the gradient:
    V_b = (var_s+ epsilon) ** (-0.5)
    part_1 = g * V_b

    # Second part pf the gradient
    diff = s - mean_s
    grad_J_vb = -0.5 * np.sum(g * (var_s+epsilon) ** (-1.5) * diff, axis=1)
    grad_J_vb = np.expand_dims(grad_J_vb, axis=1)
    part_2 = (2/float(s.shape[1])) * grad_J_vb * diff

    # Third part of the gradient
    grad_J_mb = -np.sum(g * V_b, axis=1)
    grad_J_mb = np.expand_dims(grad_J_mb, axis=1)
    part_3 = grad_J_mb / float(s.shape[1])

    return part_1 + part_2 + part_3


# In[82]:


def BackwardPassBatchNormalization(X, Y, weights, biases, p, bn_outputs, bn_activations, intermediate_outputs, means, variances, regularization_term):

    # Back-propagate output layer at first

    g = p - Y

    bias_updates = [g.sum(axis=1).reshape(biases[-1].shape)]
    weight_updates = [np.dot(g, bn_activations[-1].T)]

    g = np.dot(g.T, weights[-1])
    ind = 1 * (bn_outputs[-1] > 0)
    g = g.T * ind

    for i in reversed(range(len(weights) -1)):
    # Back-propagate the gradient vector g to the layer before

        g = BatchNormBackPass(g, intermediate_outputs[i], means[i], variances[i])

        if i == 0:
            weight_updates.append(np.dot(g, X.T))
            bias_updates.append(np.sum(g, axis=1).reshape(biases[i].shape))
            break
        else:
            weight_updates.append(np.dot(g, bn_activations[i-1].T))
            bias_updates.append(np.sum(g, axis=1).reshape(biases[i].shape))

        g = np.dot(g.T, weights[i])
        ind = 1 * (bn_outputs[i-1] > 0)
        g = g.T * ind


    for elem in weight_updates:
        elem /= X.shape[1]

    for elem in bias_updates:
        elem /= X.shape[1]

    # Reverse the updates to match the order of the layers
    weight_updates = list(reversed(weight_updates)).copy()
    bias_updates = list(reversed(bias_updates)).copy()

    for index in range(len(weight_updates)):
        weight_updates[index] += 2*regularization_term * weights[index]

    return weight_updates, bias_updates


# In[83]:


def ComputeGradsNumSlowBatchNorm(X, Y, weights, biases, start_index=0, h=1e-5):


    grad_weights = []
    grad_biases = []

    for layer_index in range(start_index, len(weights)):

        W = weights[layer_index]
        b = biases[layer_index]

        grad_W = np.zeros(W.shape)
        grad_b = np.zeros(b.shape)

        for i in tqdm(range(b.shape[0])):
            b_try = np.copy(b)
            b_try[i, 0] -= h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c1, _ = ComputeCostBatchNormalization(X=X, Y=Y, weights=weights, biases=temp_biases, regularization_term=0)
            b_try = np.copy(b)
            b_try[i, 0] += h
            temp_biases = biases.copy()
            temp_biases[layer_index] = b_try
            c2, _ = ComputeCostBatchNormalization(X=X, Y=Y, weights=weights, biases=temp_biases, regularization_term=0)

            grad_b[i, 0] = (c2 - c1) / (2 * h)

        grad_biases.append(grad_b)

        for i in tqdm(range(W.shape[0])):
            for j in range(W.shape[1]):
                W_try = np.copy(W)
                W_try[i, j] -= h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c1, _ = ComputeCostBatchNormalization(X=X, Y=Y, weights=temp_weights, biases=biases, regularization_term=0)
                W_try = np.copy(W)
                W_try[i, j] += h
                temp_weights = weights.copy()
                temp_weights[layer_index] = W_try
                c2, _ = ComputeCostBatchNormalization(X=X, Y=Y, weights=temp_weights, biases=biases, regularization_term=0)

                grad_W[i, j] = (c2 - c1) / (2 * h)

        grad_weights.append(grad_W)

    return grad_weights, grad_biases


# In[84]:


W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072), (10, 50)])


X_train, Y_train, y_train = load('data_batch_1')
X_validation, Y_validation, y_validation = load('data_batch_2')
X_test, Y_test, y_test = load('test_batch')

X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)


# In[85]:


#Layer 2

p, batch_normalization_activations, batch_normalization_outputs, intermediate_outputs, means, variances = ForwardPassBatchNormalization(X_train[:, :100], W, b)
grad_W, grad_b = BackwardPassBatchNormalization(X_train[:, :100], Y_train[:, :100], W, b, p, batch_normalization_outputs, batch_normalization_activations, intermediate_outputs, means, variances, regularization_term=0)



# In[86]:


grad_W_num, grad_b_num = ComputeGradsNumSlowBatchNorm(X_train[:, :100], Y_train[:, :100], W, b, start_index=0, h=1e-5)


# In[59]:


ComputeTestingGradient(grad_W, grad_b, grad_W_num, grad_b_num)


# In[60]:


# layer 3

W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072),(50, 50), (10, 50)])


X_train, Y_train, y_train = load('data_batch_1')
X_validation, Y_validation, y_validation = load('data_batch_2')
X_test, Y_test, y_test = load('test_batch')

X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)


# In[61]:


p, batch_normalization_activations, batch_normalization_outputs, intermediate_outputs, means, variances = ForwardPassBatchNormalization(X_train[:, :100], W, b)
grad_W, grad_b = BackwardPassBatchNormalization(X_train[:, :100], Y_train[:, :100], W, b, p, batch_normalization_outputs, batch_normalization_activations, intermediate_outputs, means, variances, regularization_term=0)




# In[62]:


grad_W_num, grad_b_num = ComputeGradsNumSlowBatchNorm(X_train[:, :100], Y_train[:, :100], W, b, start_index=0, h=1e-5)



# In[63]:


ComputeTestingGradient(grad_W, grad_b, grad_W_num, grad_b_num)


# In[64]:


def ExponentialMovingAverage(means, exponential_means, variances, exponential_variances, a=0.99):

    for index, elem in enumerate(exponential_means):

        exponential_means[index] = a * elem + (1-a) * means[index]
        exponential_variances[index] = a * exponential_variances[index] + (1-a) * variances[index]

    return exponential_means, exponential_variances


# In[75]:


def randomize_data_order(data_x, data_y, labels):
    import numpy as np

    idx = np.arange(data_x.shape[1])
    np.random.shuffle(idx)
    return data_x[:, idx], data_y[:, idx], labels[idx]

def train_model_bn(W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size):
    training_cost = []
    training_loss = []
    training_accuracy = []
    validation_cost = []
    validation_accuracy = []
    validation_loss = []
    X_train, Y_train, y_train = dataset["train"]
    X_val, Y_val, y_val = dataset["val"]
    num_steps = X_train.shape[1] / batch_size
    #print(X_train.shape[1])
    #num_steps = (5*45000) / batch_size


    clr = CyclicalLearningRate(min_lr, max_lr, n_epochs, num_steps)
    lr = min_lr

    while clr.current_cycle < 3:
        # Randomize the data
        X_train, Y_train, y_train = randomize_data_order(X_train, Y_train, y_train)

        #for batch_idx in tqdm(range(int(n_epochs))):
        for batch_idx in tqdm(range(int(num_steps))):
            X_batch = X_train[:, batch_idx * batch_size : (batch_idx + 1) * batch_size]
            Y_batch = Y_train[:, batch_idx * batch_size : (batch_idx + 1) * batch_size]
            p, batch_normalization_activations, batch_normalization_outputs, intermediate_outputs, means, variances = ForwardPassBatchNormalization(
                X_batch, W, b)

            grad_W, grad_b = BackwardPassBatchNormalization(
                X_batch, Y_batch, W, b, p, batch_normalization_outputs, batch_normalization_activations, intermediate_outputs, means, variances, regularization_term=lamda)
    
    
            if batch_idx == 0:
                exponential_means = means.copy()
                exponential_variances = variances.copy()
                exponentials, best_exponentials = [exponential_means, exponential_variances], [exponential_means, exponential_variances]
            else:
                exponentials = ExponentialMovingAverage(means, exponentials[0], variances, exponentials[1])
            # Update weights and biases
            W = [W[layer] - lr*grad_W[layer] for layer in range(len(W))]
            b = [b[layer] - lr*grad_b[layer] for layer in range(len(b))]
        
            #gamma = [gamma[layer] - lr*grad_gamma[layer] for layer in range(len(gamma))]
            #beta = [beta[layer] - lr*grad_beta[layer] for layer in range(len(beta))]
            
            lr = clr()
        training_cost.append(ComputeCostBatchNormalization(X_train, Y_train, W, b, lamda, exponentials=exponentials)[0])
        training_loss.append(ComputeCostBatchNormalization(X_train, Y_train, W, b, lamda, exponentials=exponentials)[1])
        training_accuracy.append(ComputeAccuracyBatchNormalization(X_train, y_train, W, b))
        validation_cost.append(ComputeCostBatchNormalization(X_val, Y_val, W, b, lamda, exponentials=exponentials)[0])
        validation_loss.append(ComputeCostBatchNormalization(X_val, Y_val, W, b, lamda, exponentials=exponentials)[1])
        validation_accuracy.append(ComputeAccuracyBatchNormalization(X_val, y_val, W, b))

    return W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost



# In[76]:


# layer 3

W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072),(50, 50), (10, 50)])


X_train, Y_train, y_train = load('data_batch_1')
X_validation, Y_validation, y_validation = load('data_batch_2')
X_test, Y_test, y_test = load('test_batch')

X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)


# In[77]:


min_lr, max_lr, lamda, n_epochs, batch_size = 1e-5, 1e-2, 0.005, 500, 100
dataset = load_dataset()


# In[89]:


W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost= train_model_bn(
    W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)


# In[90]:


plt.plot(training_cost, label='Training cost')
plt.plot(validation_cost, label='Validation cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.show()

plt.plot(training_loss, label='Training loss')
plt.plot(validation_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(training_accuracy, label='Training accuracy')
plt.plot(validation_accuracy, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[93]:


def generate_lambda_range(lower_limit, upper_limit):
    import numpy as np

    return np.logspace(lower_limit, upper_limit, num=10, base=10)

def refine_lambda_search(final_lambda):
    import numpy as np

    return np.linspace(
        final_lambda - final_lambda * 1e-1, final_lambda + final_lambda * 1e-1, num=10
    )


# In[96]:


low_lamda = -5
high_lamda = -1
lamdas = generate_lambda_range(low_lamda, high_lamda)
optimal_lamda = 0
best_accuracy = float("-inf")
lamda_accuracy = {}
lamda_accuracy[0] = float("-inf")

W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072),(50, 50), (10, 50)])


X_train, Y_train, y_train = load('data_batch_1')
X_validation, Y_validation, y_validation = load('data_batch_2')
X_test, Y_test, y_test = load('test_batch')

X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)
min_lr, max_lr, lamda, n_epochs, batch_size = 1e-5, 1e-2, 0.005, 500, 100
dataset = load_dataset()
for lamda in tqdm(lamdas):
    #print(f"Grid Search Lambda: {lambda_val}")
    (
        W,
        b,
        training_loss,
        training_accuracy,
        validation_loss,
        validation_accuracy,
        training_cost,
        validation_cost
    ) = train_model_bn(W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)

    lamda_accuracy[lamda] = np.max(validation_accuracy)

optimal_lamda = max(lamda_accuracy, key=lamda_accuracy.get)


# In[97]:


print(sorted(lamda_accuracy.items(), key=lambda item: item[1], reverse=True))


# In[98]:


optimal_lamda = max(lamda_accuracy, key=lamda_accuracy.get)


# In[99]:


print(optimal_lamda)



# In[101]:


refined_lamda_accuracy = {}
refined_lamda_accuracy[0] = float("-inf")

refined_lamdas = refine_lambda_search(optimal_lamda)
for lamda in refined_lamdas:
    #print(f"Fine Search Lambda: {lambda_val}")
    (
        W,
        b,
        training_loss,
        training_accuracy,
        validation_loss,
        validation_accuracy,
        training_cost,
        validation_cost
    ) = train_model_bn(W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)

    refined_lamda_accuracy[lamda] = np.max(validation_accuracy)

best_refined_lamdda = max(refined_lamda_accuracy, key=refined_lamda_accuracy.get)


# In[102]:


print(sorted(refined_lamda_accuracy.items(), key=lambda item: item[1], reverse=True))


# In[112]:


# 9 layers with BN

X_train, Y_train, y_train = load('data_batch_1')
X_validation, Y_validation, y_validation = load('data_batch_2')
X_test, Y_test, y_test = load('test_batch')

X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)
min_lr, max_lr, lamda, n_epochs, batch_size = 1e-5, 1e-2, 0.005, 500, 100
dataset = load_dataset()
#W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072),(50, 30, 20, 20, 10, 10, 10, 10), (10, 50)])
W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)])


# In[113]:


W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost= train_model_bn(
    W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)


# In[114]:


plt.plot(training_cost, label='Training cost')
plt.plot(validation_cost, label='Validation cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.show()

plt.plot(training_loss, label='Training loss')
plt.plot(validation_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(training_accuracy, label='Training accuracy')
plt.plot(validation_accuracy, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[127]:


# 9 layers without BN

X_train, Y_train, y_train = load('data_batch_1')
X_validation, Y_validation, y_validation = load('data_batch_2')
X_test, Y_test, y_test = load('test_batch')

X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)
min_lr, max_lr, lamda, n_epochs, batch_size = 1e-5, 1e-2, 0.005, 500, 100
dataset = load_dataset()
#W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072),(50, 30, 20, 20, 10, 10, 10, 10), (10, 50)])
W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072), (30, 50), (20, 30), (20, 20), (10, 20), (10, 10), (10, 10), (10, 10), (10, 10)])



# In[128]:


W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost= train_model(
    W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)


# In[129]:


plt.plot(training_cost, label='Training cost')
plt.plot(validation_cost, label='Validation cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.show()

plt.plot(training_loss, label='Training loss')
plt.plot(validation_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(training_accuracy, label='Training accuracy')
plt.plot(validation_accuracy, label='Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# In[130]:


def initialize_weights_and_bn_params(shapes_list, sigma):
    np.random.seed(369)

    weights = []
    biases = []
    gammas = []
    betas = []

    for shape in shapes_list:
        W = np.random.normal(0, sigma, size=(shape[0], shape[1]))
        b = np.zeros(shape=(shape[0], 1))
        gamma = np.ones((shape[0], 1))
        beta = np.zeros((shape[0], 1))

        weights.append(W)
        biases.append(b)
        gammas.append(gamma)
        betas.append(beta)

    return weights, biases, gammas, betas


# In[133]:


sigmas = [1e-1, 1e-3, 1e-4]

[(50, 3072),(50, 50), (10, 50)]
X_train, Y_train, y_train = load('data_batch_1')
X_validation, Y_validation, y_validation = load('data_batch_2')
X_test, Y_test, y_test = load('test_batch')

X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)
min_lr, max_lr, lamda, n_epochs, batch_size = 1e-5, 1e-2, 0.005, 500, 100
dataset = load_dataset()
W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost= train_model_bn(
    W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)

for sigma in sigmas:
    # initialize weights and biases with the given sigma
    W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072),(50, 50), (10, 50)], sigma)

    # train the network with batch normalization
    W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost= train_model_bn(
    W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)
    plt.plot(training_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.title('Training loss with BN, sigma = {}'.format(sigma))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print(f'Training accuracy with BN: {training_accuracy}, Sigma: {sigma}')
    # re-initialize weights and biases with the same sigma
    W, b, gamma, beta = initialize_weights_and_bn_params([(50, 3072),(50, 50), (10, 50)], sigma)
    

    # train the network without batch normalization
    W, b, training_loss, training_accuracy, validation_loss, validation_accuracy, training_cost, validation_cost= train_model(
    W, b, dataset, min_lr, max_lr, lamda, n_epochs, batch_size)
    
    plt.plot(training_loss, label='Training loss')
    plt.plot(validation_loss, label='Validation loss')
    plt.title('Training loss without BN, sigma = {}'.format(sigma))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print(f'Training accuracy without BN: {training_accuracy}, Sigma: {sigma}')




# In[ ]:




