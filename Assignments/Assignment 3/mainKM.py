import numpy as np
import pickle
import matplotlib.pyplot as plt
import random

def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def load_batch(filename):
    """ Copied from the dataset website """
    with open('C:/Users/krist/OneDrive/Skrivbord/DD2424/Datasets/'+filename, 'rb') as fo:
        dataset_dict = pickle.load(fo, encoding='bytes')
 
    return dataset_dict

def compute_grads_num(X, Y, lamda, W, b, gamma, beta, h, mean=None, var=None, batch_norm=False):
    
    # Initialize gradients by layers
    grad_W = [np.zeros_like(W_l) for W_l in W]
    grad_b = [np.zeros_like(b_l) for b_l in b]
    if batch_norm:
        grad_gamma = [np.zeros_like(gamma_l) for gamma_l in gamma]
        grad_beta = [np.zeros_like(beta_l) for beta_l in beta]
    
    # Compute initial cost and iterate layers k
    c, _ = compute_cost(X, Y, lamda, W, b, gamma, beta, mean, var, batch_norm)
    k = len(W)
    for layer in range(k):
        
        # Gradients for bias
        for i in range(b[layer].shape[0]):
            b_try = [np.copy(b_l) for b_l in b]
            b_try[layer][i,0] += h
            c2, _ = compute_cost(X, Y, lamda, W, b_try, gamma, beta, mean, var, batch_norm)
            grad_b[layer][i,0] = (c2-c)/h
        
        # Gradients for weights
        for i in range(W[layer].shape[0]):
            for j in range(W[layer].shape[1]):
                W_try = [np.copy(W_l) for W_l in W]
                W_try[layer][i,j] += h
                c2, _ = compute_cost(X, Y, lamda, W_try, b, gamma, beta, mean, var, batch_norm)
                grad_W[layer][i,j] = (c2-c)/h
                
        if layer<(k-1) and batch_norm:
            # Gradients for gamma
            for i in range(gamma[layer].shape[0]):
                gamma_try = [np.copy(gamma_l) for gamma_l in gamma]
                gamma_try[layer][i,0] += h
                c2, _ = compute_cost(X, Y, lamda, W, b, gamma_try, beta, mean, var, batch_norm)
                grad_gamma[layer][i,0] = (c2-c)/h
            
            # Gradients for betas
            for i in range(beta[layer].shape[0]):
                beta_try = [np.copy(beta_l) for beta_l in beta]
                beta_try[layer][i,0] += h
                c2, _ = compute_cost(X, Y, lamda, W, b, gamma, beta_try, mean, var, batch_norm)
                grad_beta[layer][i,0] = (c2-c)/h
    
    if batch_norm:
        return grad_W, grad_b, grad_gamma, grad_beta
    else:
        return grad_W, grad_b

def load_dataset(filename):
    # Load the dataset
    dataset = load_batch(filename)
    
    X = dataset[b'data'].astype(np.float32).reshape(-1, 3072).T / 255
    y = np.array(dataset[b'labels']).astype(np.int32)
    
    Y = np.zeros((10, y.shape[0]), dtype=np.float32)
    for i, label in enumerate(y):
        Y[label, i] = 1
    
    return X, Y, y + 1

def normalize(train, validation, test):
    # Compute the mean and standard deviation of the training data
    train_mean = np.mean(train, axis=1, keepdims=True)
    train_std = np.std(train, axis=1, keepdims=True)

    # Normalize the training, validation and test data
    train_norm = (train - train_mean) / train_std
    validation_norm = (validation - train_mean) / train_std
    test_norm = (test - train_mean) / train_std

    return train_norm, validation_norm, test_norm

def init_params(input_dim, hidden_dim, output_dim, seed=42, he=True, sigma=None):
    np.random.seed(seed)
    k = len(hidden_dim) + 1  # k layers
    nodes = [input_dim] + hidden_dim + [output_dim]

    def initialize_weights(n_in, n_out):
        if sigma is not None:
            return np.random.normal(0, sigma, size=(n_out, n_in))
        elif he is True:
            return np.random.normal(0, np.sqrt(2 / n_in), size=(n_out, n_in))
        else:
            return np.random.normal(0, np.sqrt(1 / n_in), size=(n_out, n_in))

    W = [initialize_weights(nodes[i], nodes[i+1]) for i in range(k)]
    b = [np.zeros((nodes[i+1], 1)) for i in range(k)]

    if he is True:
        gamma = [np.sqrt(2/nodes[i+1])*np.ones((nodes[i+1], 1)) for i in range(k-1)]
    else:
        gamma = [np.ones((nodes[i+1], 1)) for i in range(k-1)]

    beta = [np.zeros((nodes[i+1], 1)) for i in range(k-1)]

    return W, b, gamma, beta

def relu(S):
    H = S
    H[H<0] = 0
    return H

def forward_pass(X, W, b, gamma=None, beta=None, mean=None, var=None, batch_norm=False):
    
    num_layers = len(W)
    X_layer =[X.copy()]+[None]*(num_layers-1)
    S = [None]*(num_layers-1)
    S_batch_norm = [None]*(num_layers-1) 
    init_vm = False
    
    if batch_norm is True:
        if mean is None and var is None:
            init_vm = True
            mean, var = [None]*(num_layers-1), [None]*(num_layers-1)
    
    for layer in range(num_layers-1):
       
        S[layer] = np.matmul(W[layer], X_layer[layer]) + b[layer]
        if batch_norm is True:
            if init_vm is True:
                mean[layer] = S[layer].mean(axis=1).reshape(-1,1)
                var[layer] = S[layer].var(axis=1).reshape(-1,1)
            S_batch_norm[layer] = (S[layer]-mean[layer])/(np.sqrt(var[layer]+1e-15))
            S_BatchNorm_Scaled = S_batch_norm[layer]*gamma[layer]+beta[layer]
            X_layer[layer+1] = relu(S_BatchNorm_Scaled)
        else:
            X_layer[layer+1] = relu(S[layer])
            
    P = softmax(np.matmul(W[num_layers-1], X_layer[num_layers-1]) + b[num_layers-1])
                          
    if batch_norm is True and init_vm is True:
        return P, S_batch_norm, S, X_layer, mean, var
    elif batch_norm is True:
        return P, S_batch_norm, S, X_layer
    else:
        return P, X_layer

def backward_pass(X, Y, P, S_batch_norm, S, X_layer, W, b, lamda, gamma=None, beta=None, mean=None, var=None, batch_norm=False):
    num_layers = len(W)
    
    # initialize gradients
    grad_W = [None] * num_layers
    grad_b = [None] * num_layers
    grad_gamma = [None] * (num_layers - 1) if batch_norm else None
    grad_beta = [None] * (num_layers - 1) if batch_norm else None
    
    # calculate gradients for output layer
    G = P - Y
    grad_W[num_layers-1] = np.matmul(G, X_layer[num_layers-1].T) / X.shape[1] + 2 * lamda * W[num_layers-1]
    grad_b[num_layers-1] = np.matmul(G, np.ones((X.shape[1], 1))) / X.shape[1]
    
    # backpropagate gradients through hidden layers
    for layer in range(num_layers-2, -1, -1):
        G = np.matmul(W[layer+1].T, G) * (X_layer[layer+1] > 0)
        
        if batch_norm:
            grad_gamma[layer] = np.matmul((G * S_batch_norm[layer]), np.ones((X.shape[1], 1))) / X.shape[1]
            grad_beta[layer] = np.matmul(G, np.ones((X.shape[1], 1))) / X.shape[1]
            G = batch_norm_back_pass(G, S[layer], mean[layer], var[layer]) * gamma[layer]
        
        grad_W[layer] = np.matmul(G, X_layer[layer].T) / X.shape[1] + 2 * lamda * W[layer]
        grad_b[layer] = np.matmul(G, np.ones((X.shape[1], 1))) / X.shape[1]
    
    if batch_norm:
        return grad_W, grad_b, grad_gamma, grad_beta
    else:
        return grad_W, grad_b
    
def batch_norm_back_pass(G, S, mean, var):
    m = S.shape[1]
    dS = S - mean
    inv_std = 1 / np.sqrt(var + 1e-15)

    G_1 = G * inv_std
    G_2 = G * inv_std ** 3

    d_mean = -np.sum(G_1, axis=1, keepdims=True)
    d_var = -0.5 * np.sum(G_2 * dS, axis=1, keepdims=True)

    dS = G_1 + 2 * d_var * dS / m + d_mean / m

    return dS

def compute_accuracy(X, y, W, b, gamma=None, beta=None, mean=None, var=None, batch_norm=False):
    if batch_norm is True and mean is None and var is None:
        P, _, _, _, mean, var = forward_pass(X, W, b, gamma, beta, batch_norm=True)
    elif batch_norm is True:
        P, _, _, _ = forward_pass(X, W, b, gamma, beta, mean, var, batch_norm=True)
    else:
        P, _ = forward_pass(X, W, b)

    # Compute accuracy
    y_preds = np.argmax(P, axis=0) + 1
    accuracy = np.mean(y_preds == y)

    return accuracy

def compute_cost(X, Y, lamda, W, b, gamma=None, beta=None, mean=None, var=None, batch_norm=False):
    if batch_norm is True and mean is None and var is None:
        P, _, _, _, mean, var = forward_pass(X, W, b, gamma, beta, batch_norm=True)
    elif batch_norm is True:
        P, _, _, _ = forward_pass(X, W, b, gamma, beta, mean, var, batch_norm=True)
    else:
        P, _ = forward_pass(X, W, b)

    # Compute the loss function term
    loss = -np.sum(Y*np.log(P)) / X.shape[1]

    # Compute the regularization term
    reg = sum([lamda*np.sum(W_l**2) for W_l in W])

    # compute the cost
    cost = loss + reg
    
    return cost, loss


filename = "cifar-10-batches-py/data_batch_1"
filename2 = "cifar-10-batches-py/data_batch_2"
filename3 = "cifar-10-batches-py/data_batch_3"
filename4 = "cifar-10-batches-py/data_batch_4"
filename5 = "cifar-10-batches-py/data_batch_5"
filename6 = "cifar-10-batches-py/test_batch"

X, Y, y = load_dataset(filename)
X2, Y2, y2 = load_dataset(filename2)
X3, Y3, y3 = load_dataset(filename3)
X4, Y4, y4 = load_dataset(filename4)
X5, Y5, y5 = load_dataset(filename5)
X_test, Y_test, y_test = load_dataset(filename6)

X_train = np.concatenate((X, X2, X3, X4, X5), axis=1)
Y_train = np.concatenate((Y, Y2, Y3, Y4, Y5), axis=1)
y_train = np.concatenate((y, y2, y3, y4, y5), axis=0)

X_validation = X_train[:, :5000]
Y_validation = Y_train[:, :5000]
y_validation = y_train[:5000]
X_train = X_train[:, 5000:]
Y_train = Y_train[:, 5000:]
y_train = y_train[5000:]

X_train_norm, X_validation_norm, X_test_norm = normalize(X_train, X_validation, X_test)

# X_train_norm = d x n
# m = 50
# k = 10

#EXCERSICE 1
# Select a small portion of the data
#X_small = X_train_norm[:10, :10]
#Y_small = Y_train[:, 0:10]
#lamda = 0
#sigma = 1e-4
W, b, gamma, beta = init_params(X_train_norm.shape[0], [50, 50], Y_train.shape[0])

"""
# Compute the analytical gradients
P_small, S_batch_norm, S, X_layer, mean, var = forward_pass(X_small, W, b, gamma=gamma, beta=beta, mean=None, var=None, batch_norm=True)

grad_W_analytical, grad_b_analytical, grad_gamma_analytical, grad_beta_analytical = backward_pass(X_small, Y_small,
                                            P_small, S_batch_norm=S_batch_norm, S=S, X_layer=X_layer, W=W, b=b, lamda=lamda, gamma=gamma,
                                            beta=beta, mean=mean, var=var, batch_norm=True)

# Compute the numerical gradients
grad_W_numerical, grad_b_numerical, grad_gamma_numerical, grad_beta_numerical = compute_grads_num(X_small, Y_small,
                                            lamda, W, b, gamma=gamma, beta=beta, h=1e-5, mean=None, var=None, batch_norm=True)

# Compute the relative error
def relative_error(grad_analytical, grad_numerical):
    return np.abs(grad_analytical-grad_numerical) / np.maximum(1e-6, np.abs(grad_analytical)+np.abs(grad_numerical))

for i in range(len(grad_W_analytical)):
    print("Relative error for W" + str(i) + ":", np.mean(relative_error(grad_W_analytical[i], grad_W_numerical[i])))
    print("Relative error for b" + str(i) + ":", np.mean(relative_error(grad_b_analytical[i], grad_b_numerical[i])))

for i in range(len(grad_gamma_analytical)):
    print("Relative error for gamma" + str(i) + ":", np.mean(relative_error(grad_gamma_analytical[i], grad_gamma_numerical[i])))
    print("Relative error for beta" + str(i) + ":", np.mean(relative_error(grad_beta_analytical[i], grad_beta_numerical[i])))
"""


#EXCERSICE 2
def mini_batch_gradient_descent_cyclical(X, Y, y_train, X_val, Y_val, y_val, W, b, lamda, GD_params, alpha, gamma=None, beta=None, batch_norm=False):
    n = X.shape[1]
    n_batch, eta_min, eta_max, n_s, n_epochs = GD_params
    train_cost = []
    val_cost = []
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []
    t = 0

    for n_epochs in range(n_epochs):

        ind = np.random.permutation(X.shape[1])
        X = X[:, ind]
        Y = Y[:, ind]
        y_train = [y_train[index] for index in ind]

        for j in range(n // n_batch):
            t = t % (2 * n_s)  # one full cycle is 2*n_s
            # cyclic learning rate
            if t <= n_s:
                eta = eta_min + (t / n_s) * (eta_max - eta_min)
            else:
                eta = eta_max - ((t - n_s) / n_s) * (eta_max - eta_min)

            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            indices = slice(j_start, j_end)
            X_batch = X[:, indices]
            Y_batch = Y[:, indices]

            if batch_norm is True:
                p_batch, S_batch_norm, S, X_layer, mean, var = forward_pass(X_batch, W, b, gamma=gamma, beta=beta, batch_norm=True)
            else:
                p_batch, X_layer = forward_pass(X_batch, W, b)

            # Setting average mean and var
            if n_epochs == 0 and j == 0 and batch_norm is True:
                mean_2 = mean
                var_2 = var
            elif batch_norm is True:
                mean_2 = [alpha * mean_2[l] + (1-alpha) * mean[l] for l in range(len(mean))]
                var_2 = [alpha * var_2[l] + (1-alpha) * var[l] for l in range(len(var))]
            else:
                mean_2 = None
                var_2 = None

            if batch_norm is True:
                grad_W, grad_b, grad_gamma, grad_beta = backward_pass(X_batch, Y_batch, p_batch,
                                                    S_batch_norm=S_batch_norm, S=S, X_layer=X_layer, W=W, b=b,
                                                    lamda=lamda, gamma=gamma, beta=beta, mean=mean_2, var=var_2,
                                                    batch_norm=True)
            else:
                grad_W, grad_b = backward_pass(X_batch, Y_batch, p_batch, S_batch_norm=None, S=None, X_layer=X_layer, W=W, b=b, lamda=lamda)

            # Update weights and biases
            W = [W[layer] - eta*grad_W[layer] for layer in range(len(W))]
            b = [b[layer] - eta*grad_b[layer] for layer in range(len(b))]
            # Update gamma and beta
            if batch_norm is True:
                gamma = [gamma[layer] - eta*grad_gamma[layer] for layer in range(len(gamma))]
                beta = [beta[layer] - eta*grad_beta[layer] for layer in range(len(beta))]

            # store the cost, loss and accuracy after a number of iterations
            if t % 450 == 0:

                train_cost.append(compute_cost(X, Y, lamda, W, b, gamma, beta, mean_2, var_2, batch_norm)[0])
                val_cost.append(compute_cost(X_val, Y_val, lamda, W, b, gamma, beta, mean_2, var_2, batch_norm)[0])

                train_loss.append(compute_cost(X, Y, lamda, W, b, gamma, beta, mean_2, var_2, batch_norm)[1])
                val_loss.append(compute_cost(X_val, Y_val, lamda, W, b, gamma, beta, mean_2, var_2, batch_norm)[1])

                train_accuracy.append(compute_accuracy(X, y_train, W, b, gamma, beta, mean_2, var_2, batch_norm))
                val_accuracy.append(compute_accuracy(X_val, y_val, W, b, gamma, beta, mean_2, var_2, batch_norm))

            t += 1

    return W, b, train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy


n = X_train_norm.shape[1]
n_batch = 100
eta_min = 1e-5
eta_max = 1e-1
n_s = (5 * 45000) / n_batch
n_epochs = 30
lamda = 0.00205
alpha = 0.9
GD_params = (n_batch, eta_min, eta_max, n_s, n_epochs)

W, b, train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy = mini_batch_gradient_descent_cyclical(
    X_train_norm, Y_train, y_train,
    X_validation_norm, Y_validation, y_validation,
    W, b, lamda, GD_params, alpha, gamma, beta, batch_norm=True)

print("Final accuracy on validation set:", val_accuracy[-1])

# Plot the training and validation loss
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot the training and validation cost
plt.plot(train_cost, label='Training cost')
plt.plot(val_cost, label='Validation cost')
plt.xlabel('Updates')
plt.ylabel('Cost')
plt.legend()
plt.show()

# Plot the training and validation accuracy
plt.plot(train_accuracy, label='Training accuracy')
plt.plot(val_accuracy, label='Validation accuracy')
plt.xlabel('Updates')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


"""
#EXCERCISE 3
n_batch = 100
n = X_train_norm.shape[1]
n_s = (5 * 45000) / n_batch
eta_min = 1e-5
eta_max = 1e-1
n_epochs = 20
alpha = 0.9
lamda_range = (1e-5, 1e-2)
GD_params = (n_batch, eta_min, eta_max, n_s, n_epochs)

def random_search(X_train_norm, Y_train, y_train,
        X_validation_norm, Y_validation, y_validation,
        W, b, GD_params, alpha, gamma, beta, lamda_range):

    lamda_performances = []

    for i in range(8):
        print("Iteration:", i+1)
        lamda_min, lamda_max = lamda_range
        lamda = random.uniform(lamda_min, lamda_max)

        W, b, train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy = mini_batch_gradient_descent_cyclical(
        X_train_norm, Y_train, y_train,
        X_validation_norm, Y_validation, y_validation,
        W, b, lamda, GD_params, alpha, gamma, beta, batch_norm=True)

        lamda_performances.append((lamda, val_accuracy[-1]))

    return lamda_performances

lamda_performances = random_search(X_train_norm, Y_train, y_train,
        X_validation_norm, Y_validation, y_validation,
        W, b, GD_params, alpha, gamma, beta, lamda_range)

# Sort the performances
lamda_performances.sort(key=lambda x: x[1], reverse=True)

# Print the performances
for lamda, performance in lamda_performances:
    print("Lambda: {:.5f}, Accuracy: {:.5f}".format(lamda, performance))
"""