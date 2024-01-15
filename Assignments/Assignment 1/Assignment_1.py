import numpy as np
import pickle
import matplotlib.pyplot as plt
 
 #From updated functions.py from Canvas. Changed names on functions from previous version.
 
def softmax(x):
    """ Standard definition of the softmax function """
    return np.exp(x) / np.sum(np.exp(x), axis=0)
 
 
def LoadBatch(filename):
    """ Copied from the dataset website """
    with open('Datasets/' + filename, 'rb') as fo:
        dataset_dict = pickle.load(fo, encoding='bytes')
 
    return dataset_dict
 
 
def ComputeGradsNum(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    # d = X.shape[0]
 
    grad_w = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))
 
    c = ComputeCost(X, Y, W, b, lamda)
 
    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
        grad_b[i] = (c2 - c) / h
 
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w_try = np.array(W)
            w_try[i, j] += h
            c2 = ComputeCost(X, Y, w_try, b, lamda)
            grad_w[i, j] = (c2 - c) / h
 
    return [grad_w, grad_b]
 
 
def ComputeGradsNumSlow(X, Y, P, W, b, lamda, h):
    """ Converted from matlab code """
    no = W.shape[0]
    # d = X.shape[0]
 
    grad_w = np.zeros(W.shape)
    grad_b = np.zeros((no, 1))
 
    for i in range(len(b)):
        b_try = np.array(b)
        b_try[i] -= h
        c1 = ComputeCost(X, Y, W, b_try, lamda)
 
        b_try = np.array(b)
        b_try[i] += h
        c2 = ComputeCost(X, Y, W, b_try, lamda)
 
        grad_b[i] = (c2 - c1) / (2 * h)
 
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            w_try = np.array(W)
            w_try[i, j] -= h
            c1 = ComputeCost(X, Y, w_try, b, lamda)
 
            w_try = np.array(W)
            w_try[i, j] += h
            c2 = ComputeCost(X, Y, w_try, b, lamda)
 
            grad_w[i, j] = (c2 - c1) / (2 * h)
 
    return [grad_w, grad_b]
 
 
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


# assignment 1.1

def load(filename):
    data = LoadBatch(filename)
    
    X = data[b'data'].astype(np.float32).reshape(-1, 3072).T / 255
    y = np.array(data[b'labels']).astype(np.int32)
    
    Y = np.zeros((10, y.shape[0]), dtype=np.float32)
    for i, label in enumerate(y):
        Y[label, i] = 1
    
    return X, Y, y + 1

train_X, train_Y, train_y = load('data_batch_1')

val_X, val_Y, val_y = load('data_batch_2')

test_X, test_Y, test_y = load('test_batch')


# assignment 1.2

def normalize_data(trainX, X):
    mean_X = np.mean(trainX, axis=1, keepdims=True)
    std_X = np.std(trainX, axis=1, keepdims=True)
    
    normalized_X = (X - mean_X) / std_X
    
    return normalized_X

trainX_norm = normalize_data(train_X, train_X)
valX_norm = normalize_data(train_X, val_X)
testX_norm = normalize_data(train_X, test_X)


#Assignment 1.3

def initialize_parameters(K, d):
    W = np.random.randn(K, d) * 0.01
    b = np.random.randn(K, 1) * 0.01
    return W, b

K = 10
d = trainX_norm.shape[0]
W, b = initialize_parameters(K, d)



# Assignment 1.4

def EvaluateClassifier(X, W, b):
    s = np.dot(W, X) + b
    P = softmax(s)
    return P




# Assignment 1.5

def ComputeCost(X, Y, W, b, lambda_):
    P = EvaluateClassifier(X, W, b)
    cross_entropy_loss = -np.mean(np.sum(Y * np.log(P), axis=0))
    regularization_term = lambda_ * np.sum(W**2)
    J = cross_entropy_loss + regularization_term
    return J


# Assignment 1.6

def ComputeAccuracy(X, y, W, b):
    P = EvaluateClassifier(X, W, b)
    predictions = np.argmax(P, axis=0)
    correct_predictions = np.sum(predictions == (y - 1))
    acc = correct_predictions / y.shape[0]
    return acc



# Assignment 1.7

def ComputeGradients(X, Y, P, W, lambda_):
    n = X.shape[1]
    g = P - Y
    grad_W = (1/n) * np.dot(g, X.T) + 2 * lambda_ * W
    grad_b = (1/n) * np.sum(g, axis=1).reshape(-1, 1)
    return grad_W, grad_b

# Select a small portion of the data
X_small = trainX_norm[:20, :1]
Y_small = train_Y[:, :1]
W_small = W[:, :20]
b_small = b
lambda_ = 0

# Compute the analytical gradients
P_small = EvaluateClassifier(X_small, W_small, b_small)
grad_W_analytical, grad_b_analytical = ComputeGradients(X_small, Y_small, P_small, W_small, lambda_)

# Compute the numerical gradients
grad_W_numerical, grad_b_numerical = ComputeGradsNumSlow(X_small, Y_small, P_small, W_small, b_small, lambda_, 1e-6)

# Compute the relative error
eps = 1e-6
relative_error_W = np.abs(grad_W_analytical - grad_W_numerical) / np.maximum(eps, np.abs(grad_W_analytical) + np.abs(grad_W_numerical))
relative_error_b = np.abs(grad_b_analytical - grad_b_numerical) / np.maximum(eps, np.abs(grad_b_analytical) + np.abs(grad_b_numerical))
print("Relative error W:", np.max(relative_error_W))
print("Relative error b:", np.max(relative_error_b))



# Assignment 1.8

def MiniBatchGD(X, Y, GDparams, W, b, lambda_):
    n_batch, eta, n_epochs = GDparams
    n = X.shape[1]
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = slice(j_start, j_end)
            X_batch = X[:, inds]
            Y_batch = Y[:, inds]

            P_batch = EvaluateClassifier(X_batch, W, b)
            grad_W, grad_b = ComputeGradients(X_batch, Y_batch, P_batch, W, lambda_)

            W -= eta * grad_W
            b -= eta * grad_b
        
        # Compute and print the cost after each epoch
        J_train = ComputeCost(X, Y, W, b, lambda_)
        print(f"Epoch {epoch + 1}/{n_epochs}, training cost: {J_train}")
        # Calculate the training loss and append it to train_losses
        train_loss = ComputeCost(X, Y, W, b, lambda_)
        train_losses.append(train_loss)

        # Calculate the validation loss and append it to val_losses
        val_loss = ComputeCost(val_X, val_Y, W, b, lambda_)
        val_losses.append(val_loss)

    return W, b, train_losses, val_losses

n_batch = 100
eta = 0.001
n_epochs = 40
lambda_ = 0

GDparams = (n_batch, eta, n_epochs)
W_star, b_star, train_losses, val_losses = MiniBatchGD(trainX_norm, train_Y, GDparams, W, b, lambda_)

test_accuracy = ComputeAccuracy(testX_norm, test_y, W_star, b_star)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


def plot_weights(W):
    templates = []
    for i in range(10):
        im = W[i].reshape(32, 32, 3)
        im = (im - im.min()) / (im.max() - im.min())
        templates.append(im)

    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(templates[i])
        ax.axis('off')
    plt.show()



n_batch = 200
eta = 0.001
n_epochs = 40
lambda_ = 0.1


# Plot the training and validation losses
def plot_loss(train_losses, val_losses):
    epochs = range(1, n_epochs + 1)
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, val_losses, label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


parameters = [
    {"lambda_": 0,"n_epochs": 40,  "n_batch": 100, "eta": 0.1},
    {"lambda_": 0,"n_epochs": 40,  "n_batch": 100, "eta": 0.001},
    {"lambda_": 0.1,"n_epochs": 40, "n_batch": 100, "eta": 0.001},
    {"lambda_": 1,"n_epochs": 40, "n_batch": 100, "eta": 0.001},
]
for param in parameters:
    print(f"Training with parameters: {param}")
    W, b = initialize_parameters(K, d)
    GDparams = (param["n_batch"], param["eta"], param["n_epochs"])


    W_star, b_star, train_losses, val_losses = MiniBatchGD(trainX_norm, train_Y, GDparams, W, b, lambda_=param["lambda_"])
    plot_loss(train_losses, val_losses)

    montage(W_star)
    
    test_accuracy = ComputeAccuracy(testX_norm, test_y, W_star, b_star)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%\n")