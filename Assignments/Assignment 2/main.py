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

def compute_grads_num_slow(X, Y, W1, b1, W2, b2, lamda, h):
    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros(b2.shape)

    # Gradients for b1
    for i in range(len(b1)):
        b1_try = np.array(b1)
        b1_try[i] -= h
        c1, _ = compute_cost(X, Y, W1, b1_try, W2, b2, lamda)

        b1_try = np.array(b1)
        b1_try[i] += h
        c2, _ = compute_cost(X, Y, W1, b1_try, W2, b2, lamda)

        grad_b1[i] = (c2 - c1) / (2 * h)

    # Gradients for W1
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] -= h
            c1, _ = compute_cost(X, Y, W1_try, b1, W2, b2, lamda)

            W1_try = np.array(W1)
            W1_try[i, j] += h
            c2, _ = compute_cost(X, Y, W1_try, b1, W2, b2, lamda)

            grad_W1[i, j] = (c2 - c1) / (2 * h)

    # Gradients for b2
    for i in range(len(b2)):
        b2_try = np.array(b2)
        b2_try[i] -= h
        c1, _ = compute_cost(X, Y, W1, b1, W2, b2_try, lamda)

        b2_try = np.array(b2)
        b2_try[i] += h
        c2, _ = compute_cost(X, Y, W1, b1, W2, b2_try, lamda)

        grad_b2[i] = (c2 - c1) / (2 * h)

    # Gradients for W2
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i, j] -= h
            c1, _ = compute_cost(X, Y, W1, b1, W2_try, b2, lamda)

            W2_try = np.array(W2)
            W2_try[i, j] += h
            c2, _ = compute_cost(X, Y, W1, b1, W2_try, b2, lamda)

            grad_W2[i, j] = (c2 - c1) / (2 * h)

    return grad_W1, grad_b1, grad_W2, grad_b2

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


def init_params(d, m, k):
    W1 = np.random.normal(0, 1 / np.sqrt(d), (m, d))
    W2 = np.random.normal(0, 1 / np.sqrt(m), (k, m))
    b1 = np.zeros((m, 1))
    b2 = np.zeros((k, 1))

    return W1, b1, W2, b2


def forward_pass(X, W1, b1, W2, b2):
    s1 = np.dot(W1, X) + b1
    h = np.maximum(0, s1)
    s2 = np.dot(W2, h) + b2
    p = softmax(s2)
    return p, h


def backward_pass(X, Y, P, W1, b1, W2, b2, h, lamda):
    N = X.shape[1]
    
    g = P - Y
    grad_W2 = (1 / N) * np.dot(g, h.T) + 2 * lamda * W2
    grad_b2 = (1 / N) * np.sum(g, axis=1, keepdims=True)

    g = np.dot(W2.T, g)
    g[h <= 0] = 0
    
    grad_W1 = (1 / N) * np.dot(g, X.T) + 2 * lamda * W1
    grad_b1 = (1 / N) * np.sum(g, axis=1, keepdims=True)
    
    return grad_W1, grad_b1, grad_W2, grad_b2


def compute_accuracy(X, y, W1, b1, W2, b2):
    # Compute the scores
    P, _ = forward_pass(X, W1, b1, W2, b2)

    # Compute the predicted labels
    y_pred = np.argmax(P, axis=0) + 1

    # Compute the accuracy
    accuracy = np.mean(y == y_pred)

    return accuracy


def compute_cost(X, Y, W1, b1, W2, b2, lamda):
    # Compute the scores
    P, _ = forward_pass(X, W1, b1, W2, b2)

    # Compute the loss
    loss = -np.mean(np.sum(Y * np.log(P), axis=0))

    # Compute the regularization term
    reg = lamda * (np.sum(W1 * W1) + np.sum(W2 * W2))

    # Compute the cost
    cost = loss + reg

    return cost, loss

"""
#EXCERSICE 1
filename = "cifar-10-batches-py/data_batch_1"
filename2 = "cifar-10-batches-py/data_batch_2"
filename3 = "cifar-10-batches-py/test_batch"

X_train, Y_train, y_train = load_dataset(filename)
X_validation, Y_validation, y_validation = load_dataset(filename2)
X_test, Y_test, y_test = load_dataset(filename3)
"""

#EXCERSICE 4
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

X_validation = X_train[:, :1000]
Y_validation = Y_train[:, :1000]
y_validation = y_train[:1000]
X_train = X_train[:, 1000:]
Y_train = Y_train[:, 1000:]
y_train = y_train[1000:]


X_train_norm, X_validation_norm, X_test_norm = normalize(X_train, X_validation, X_test)

# X_train_norm = d x n
# m = 50
# k = 10

W1, b1, W2, b2 = init_params(X_train_norm.shape[0], 50, Y_train.shape[0])

"""
#EXCERSICE 2
# Select a small portion of the data
X_small = X_train_norm[:, :100]
Y_small = Y_train[:, :100]
lambda_ = 0
P_small, h = forward_pass(X_small, W1, b1, W2, b2)

# Compute the analytical gradients
grad_W1_analytical, grad_b1_analytical, grad_W2_analytical, grad_b2_analytical = backward_pass(X_small, Y_small, P_small, W1, b1, W2, b2, h, lambda_)

# Compute the numerical gradients
grad_W1_numerical, grad_b1_numerical, grad_W2_numerical, grad_b2_numerical = compute_grads_num_slow(X_small, Y_small, W1, b1, W2, b2, lambda_, 1e-5)

# Compute the relative error
eps = 1e-10
relative_error_W1 = np.abs(grad_W1_analytical - grad_W1_numerical) / np.maximum(eps, np.abs(grad_W1_analytical) + np.abs(grad_W1_numerical))
relative_error_W2 = np.abs(grad_W2_analytical - grad_W2_numerical) / np.maximum(eps, np.abs(grad_W2_analytical) + np.abs(grad_W2_numerical))

relative_error_b1 = np.abs(grad_b1_analytical - grad_b1_numerical) / np.maximum(eps, np.abs(grad_b1_analytical) + np.abs(grad_b1_numerical))
relative_error_b2 = np.abs(grad_b2_analytical - grad_b2_numerical) / np.maximum(eps, np.abs(grad_b2_analytical) + np.abs(grad_b2_numerical))


print("Relative error W1:", np.max(relative_error_W1))
print("Relative error W2:", np.max(relative_error_W2))
print("Relative error b1:", np.max(relative_error_b1))
print("Relative error b2:", np.max(relative_error_b2))


def mini_batch_gd(X, Y, X_validation, Y_validation, GD_params, W1, b1, W2, b2, lambda_):
    # Initialize the parameters
    n_batch, eta, n_epochs = GD_params
    n = X.shape[1]
    train_cost = []
    validation_cost = []

    for n_epochs in range(n_epochs):
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            indices = slice(j_start, j_end)
            X_batch = X[:, indices]
            Y_batch = Y[:, indices]
            p_batch, h = forward_pass(X_batch, W1, b1, W2, b2)

            grad_W1, grad_b1, grad_W2, grad_b2 = backward_pass(X_batch, Y_batch, p_batch, W1, b1, W2, b2, h, lambda_)

            W1 -= eta * grad_W1
            b1 -= eta * grad_b1
            W2 -= eta * grad_W2
            b2 -= eta * grad_b2

        # Compute the cost
        train_cost.append(compute_cost(X, Y, W1, b1, W2, b2, lambda_)[0])
        validation_cost.append(compute_cost(X_validation, Y_validation, W1, b1, W2, b2, lambda_)[0])
	
    return W1, b1, W2, b2, train_cost, validation_cost


# sanity check
GD_params = (100, 0.1, 200)
X_validation_norm = X_validation_norm[:, :100]
Y_validation = Y_validation[:, :100]
W1, b1, W2, b2, train_cost, validation_cost = mini_batch_gd(X_small, Y_small, X_validation_norm, Y_validation, GD_params, W1, b1, W2, b2, lambda_)

print("Final training cost:", train_cost[-1])
print("Final validation cost:", validation_cost[-1])

#plot the training and validation cost
plt.plot(train_cost, label='Training cost')
plt.plot(validation_cost, label='Validation cost')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.legend()
plt.show()
"""

#EXCERSICE 3
def mini_batch_gradient_descent_cyclical(X, Y, y_train, X_val, Y_val, y_val, W1, b1, W2, b2, lamda, GD_params):
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
        print("Epoch:", n_epochs)
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
            p_batch, h = forward_pass(X_batch, W1, b1, W2, b2)

            grad_W1, grad_b1, grad_W2, grad_b2 = backward_pass(X_batch, Y_batch, p_batch, W1, b1, W2, b2, h, lamda)

            W1 -= eta * grad_W1
            b1 -= eta * grad_b1
            W2 -= eta * grad_W2
            b2 -= eta * grad_b2

            # store the cost, loss and accuracy after a number of iterations
            if t % 196 == 0:

                train_cost.append(compute_cost(X, Y, W1, b1, W2, b2, lamda)[0])
                val_cost.append(compute_cost(X_val, Y_val, W1, b1, W2, b2, lamda)[0])

                train_loss.append(compute_cost(X, Y, W1, b1, W2, b2, lamda)[1])
                val_loss.append(compute_cost(X_val, Y_val, W1, b1, W2, b2, lamda)[1])

                train_accuracy.append(compute_accuracy(X, y_train, W1, b1, W2, b2))
                val_accuracy.append(compute_accuracy(X_val, y_val, W1, b1, W2, b2))

            t += 1

    return W1, b1, W2, b2, train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy

n = X_train_norm.shape[1]
n_batch = 100
eta_min = 1e-5
eta_max = 1e-1
n_s = 2 * (n // n_batch)
n_epochs = 12
lambda_ = 0.00123
GD_params = (n_batch, eta_min, eta_max, n_s, n_epochs)

W1, b1, W2, b2, train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy = mini_batch_gradient_descent_cyclical(
    X_train_norm, Y_train, y_train,
    X_validation_norm, Y_validation, y_validation,
    W1, b1, W2, b2, lambda_, GD_params)

# Plot the training and validation loss
plt.plot(train_loss, label='Training loss')
plt.plot(val_loss, label='Validation loss')
plt.xlabel('Updates')
plt.ylabel('Loss')
plt.legend()
plt.show()


"""
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

print("Final accuracy on validation set:", val_accuracy[-1])


#EXCERCISE 4
n_batch = 100
n = X_train_norm.shape[1]
n_s = 2 * (n // n_batch)
eta_min = 1e-5
eta_max = 1e-1
n_epochs = 12
lamda_range = (1e-5, 1e-2)
GD_params = (n_batch, eta_min, eta_max, n_s, n_epochs)

def random_search(X_train_norm, Y_train, y_train,
        X_validation_norm, Y_validation, y_validation,
        W1, b1, W2, b2, GD_params, lamda_range):

    lamda_performances = []

    for i in range(8):
        lamda_min, lamda_max = lamda_range
        lamda = random.uniform(lamda_min, lamda_max)

        W1, b1, W2, b2, train_cost, val_cost, train_loss, val_loss, train_accuracy, val_accuracy = mini_batch_gradient_descent_cyclical(
        X_train_norm, Y_train, y_train,
        X_validation_norm, Y_validation, y_validation,
        W1, b1, W2, b2, lamda, GD_params)

        lamda_performances.append((lamda, val_accuracy[-1]))

    return lamda_performances

lamda_performances = random_search(X_train_norm, Y_train, y_train, X_validation_norm, Y_validation, y_validation, W1, b1, W2, b2, GD_params, lamda_range)

# Sort the performances
lamda_performances.sort(key=lambda x: x[1], reverse=True)

# Print the performances
for lamda, performance in lamda_performances:
    print("Lambda: {:.5f}, Accuracy: {:.5f}".format(lamda, performance))
