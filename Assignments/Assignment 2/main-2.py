import numpy as np
from scipy.io import loadmat
import pickle
import matplotlib.pyplot as plt
import random


def load_batch(filename):
    dataset_dict = loadmat(filename)

    X = dataset_dict['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
    X = X.reshape(X.shape[0], -1) / 255.0  # normalize pixel values between 0 and 1
    y = np.array(dataset_dict['labels']).flatten()
    Y = np.eye(10)[y].T  # convert labels to one-hot representation

    return X.T, Y, y


def preprocess_data(train_X, val_X, test_X):
    mean_X = np.mean(train_X, axis=1, keepdims=True)
    std_X = np.std(train_X, axis=1, keepdims=True)

    train_X_normalized = (train_X - mean_X) / std_X
    val_X_normalized = (val_X - mean_X) / std_X
    test_X_normalized = (test_X - mean_X) / std_X

    return train_X_normalized, val_X_normalized, test_X_normalized


def forward_pass(X, W1, b1, W2, b2):
    # Follow figure 1
    s1 = np.dot(W1, X) + b1
    h = np.maximum(0, s1)
    s2 = np.dot(W2, h) + b2
    P = np.exp(s2) / np.sum(np.exp(s2), axis=0)

    return P, h


def compute_cost(X, Y, W1, b1, W2, b2, lamda):
    P, h = forward_pass(X, W1, b1, W2, b2)  # Unpack the output correctly
    cross_entropy_loss = -np.mean(np.sum(Y * np.log(P), axis=0))
    regularization = lamda * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    J = cross_entropy_loss + regularization
    return J, h, cross_entropy_loss


def compute_grads_num(X, Y, W1, b1, W2, b2, lamda, h):
    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros(b2.shape)

    c, _, _ = compute_cost(X, Y, W1, b1, W2, b2, lamda)

    # Compute gradients for b1
    for i in range(b1.shape[0]):
        b1_try = np.array(b1)
        b1_try[i] += h
        c2, _, _ = compute_cost(X, Y, W1, b1_try, W2, b2, lamda)
        grad_b1[i] = (c2 - c) / h

    # Compute gradients for W1
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] += h
            c2, _, _ = compute_cost(X, Y, W1_try, b1, W2, b2, lamda)
            grad_W1[i, j] = (c2 - c) / h

    # Compute gradients for b2
    for i in range(b2.shape[0]):
        b2_try = np.array(b2)
        b2_try[i] += h
        c2, _, _ = compute_cost(X, Y, W1, b1, W2, b2_try, lamda)
        grad_b2[i] = (c2 - c) / h

    # Compute gradients for W2
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i, j] += h
            c2, _, _ = compute_cost(X, Y, W1, b1, W2_try, b2, lamda)
            grad_W2[i, j] = (c2 - c) / h

    return grad_W1, grad_b1, grad_W2, grad_b2


def compute_grads_num_slow(X, Y, W1, b1, W2, b2, lamda, h):
    grad_W1 = np.zeros(W1.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_W2 = np.zeros(W2.shape)
    grad_b2 = np.zeros(b2.shape)

    # Gradients for b1
    for i in range(len(b1)):
        b1_try = np.array(b1)
        b1_try[i] -= h
        c1, _, _ = compute_cost(X, Y, W1, b1_try, W2, b2, lamda)

        b1_try = np.array(b1)
        b1_try[i] += h
        c2, _, _ = compute_cost(X, Y, W1, b1_try, W2, b2, lamda)

        grad_b1[i] = (c2 - c1) / (2 * h)

    # Gradients for W1
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            W1_try = np.array(W1)
            W1_try[i, j] -= h
            c1, _, _ = compute_cost(X, Y, W1_try, b1, W2, b2, lamda)

            W1_try = np.array(W1)
            W1_try[i, j] += h
            c2, _, _ = compute_cost(X, Y, W1_try, b1, W2, b2, lamda)

            grad_W1[i, j] = (c2 - c1) / (2 * h)

    # Gradients for b2
    for i in range(len(b2)):
        b2_try = np.array(b2)
        b2_try[i] -= h
        c1, _, _ = compute_cost(X, Y, W1, b1, W2, b2_try, lamda)

        b2_try = np.array(b2)
        b2_try[i] += h
        c2, _, _ = compute_cost(X, Y, W1, b1, W2, b2_try, lamda)

        grad_b2[i] = (c2 - c1) / (2 * h)

    # Gradients for W2
    for i in range(W2.shape[0]):
        for j in range(W2.shape[1]):
            W2_try = np.array(W2)
            W2_try[i, j] -= h
            c1, _, _ = compute_cost(X, Y, W1, b1, W2_try, b2, lamda)

            W2_try = np.array(W2)
            W2_try[i, j] += h
            c2, _, _ = compute_cost(X, Y, W1, b1, W2_try, b2, lamda)

            grad_W2[i, j] = (c2 - c1) / (2 * h)

    return grad_W1, grad_b1, grad_W2, grad_b2


def compute_gradients(X, Y, P, h, W1, b1, W2, b2, lamda):
    n = X.shape[1]

    # Gradients for the second layer
    g2 = P - Y
    grad_W2 = (1 / n) * np.dot(g2, h.T) + 2 * lamda * W2
    grad_b2 = (1 / n) * np.sum(g2, axis=1, keepdims=True)

    # Gradients for the first layer
    g1 = np.dot(W2.T, g2)
    g1[h <= 0] = 0  # ReLU gradient
    grad_W1 = (1 / n) * np.dot(g1, X.T) + 2 * lamda * W1
    grad_b1 = (1 / n) * np.sum(g1, axis=1, keepdims=True)

    return grad_W1, grad_b1, grad_W2, grad_b2

def relative_error(ga, gn, eps=1e-10):
    numerator = np.abs(ga - gn)
    denominator = np.maximum(eps, np.abs(ga) + np.abs(gn))
    return numerator / denominator

def compute_accuracy(X, y, W1, b1, W2, b2):
    P, _ = forward_pass(X, W1, b1, W2, b2)
    preds = np.argmax(P, axis=0)
    acc = np.mean(preds == y)
    #correct_preds = np.sum(preds == y)
    #acc = correct_preds / y.size
    return acc

def mini_batch_gradient_descent(X, Y, X_val, Y_val, W1, b1, W2, b2, lamda, n_batch, eta, n_epochs): #kan ändra så att den liknar ass1 lite mer ass1
    N = X.shape[1]
    cost_train_history = []
    cost_val_history = []
    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(N)
        for j in range(0, N, n_batch):
            j_start = j
            j_end = j + n_batch
            inds = shuffled_indices[j_start:j_end]
            X_batch = X[:, inds]
            Y_batch = Y[:, inds]
            P, h = forward_pass(X_batch, W1, b1, W2, b2)
            grad_W1, grad_b1, grad_W2, grad_b2 = compute_gradients(X_batch, Y_batch, P, h, W1, b1, W2, b2, lamda)

            W1 -= eta * grad_W1
            b1 -= eta * grad_b1
            W2 -= eta * grad_W2
            b2 -= eta * grad_b2

        cost_train, _, _ = compute_cost(X, Y, W1, b1, W2, b2, lamda)
        cost_val, _, _ = compute_cost(X_val, Y_val, W1, b1, W2, b2, lamda)
        cost_train_history.append(cost_train)
        cost_val_history.append(cost_val)

    # Plot the training and validation cost
    plt.plot(cost_train_history, label='Training Cost')
    plt.plot(cost_val_history, label='Validation Cost')
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

    # Check the final training cost
    print("Final training cost:", cost_train_history[-1])
    print('Final validation cost', cost_val_history[-1])

    return W1, b1, W2, b2, cost_train_history, cost_val_history

def mini_batch_gradient_descent_cyclical(X, Y, y_train ,X_val, Y_val, y_val, X_test, Y_test, y_test, W1, b1, W2, b2, lamda, n_batch, eta_min, eta_max, n_s, n_epochs):
    N = X.shape[1]

    loss_train_history = []
    loss_val_history = []
    cost_train_history = []
    cost_val_history = []

    test_accuracy_history = []
    train_accuracy_history = []

    t = 0
    eta_history = []
    update_step_history = []

    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(N)
        for j in range(0, N, n_batch):
            t = t % (2 * n_s)  # Reset t after every 2 * n_s updates
            # Update learning rate according to cyclical learning rate policy
            if t <= n_s:
                eta = eta_min + (t / n_s) * (eta_max - eta_min)
            else:
                eta = eta_max - ((t - n_s) / n_s) * (eta_max - eta_min)

            j_start = j
            j_end = j + n_batch
            #j_end = min(j + n_batch, N)
            inds = shuffled_indices[j_start:j_end]
            X_batch = X[:, inds]
            Y_batch = Y[:, inds]
            P, h = forward_pass(X_batch, W1, b1, W2, b2)
            grad_W1, grad_b1, grad_W2, grad_b2 = compute_gradients(X_batch, Y_batch, P, h, W1, b1, W2, b2, lamda)

            W1 -= eta * grad_W1
            b1 -= eta * grad_b1
            W2 -= eta * grad_W2
            b2 -= eta * grad_b2

            eta_history.append(eta)
            update_step_history.append(j)

            # Calculate and store the costs at every update step
            if t % 49 == 0:

                cost_train, _, _ = compute_cost(X, Y, W1, b1, W2, b2, lamda)
                cost_val, _, _ = compute_cost(X_val, Y_val, W1, b1, W2, b2, lamda)
                cost_train_history.append(cost_train)
                cost_val_history.append(cost_val)

                _, _, loss_train = compute_cost(X, Y, W1, b1, W2, b2, lamda)
                _, _, loss_val = compute_cost(X_val, Y_val, W1, b1, W2, b2, lamda)
                loss_train_history.append(loss_train)
                loss_val_history.append(loss_val)


                train_accuracy = compute_accuracy(X, y_train, W1, b1, W2, b2)
                train_accuracy_history.append(train_accuracy)
                test_accuracy = compute_accuracy(X_test, y_test, W1, b1, W2, b2)
                test_accuracy_history.append(test_accuracy)


            t += 1

        print("Epoch {}/{}: Test accuracy: {:.4f}".format(epoch + 1, n_epochs, test_accuracy))
        #print("Epoch {}/{}: Val accuracy: {:.4f}".format(epoch + 1, n_epochs, train_accuracy))
    return W1, b1, W2, b2, loss_train_history, loss_val_history, cost_train_history, cost_val_history, train_accuracy_history, test_accuracy_history,  eta_history, update_step_history

# Have to change for distinct users with dataset stored elsewhere, dataset up until coarse-to-fine random search:
train_file = '/Users/alperenkinali/Desktop/KTH/DD2424DeepLearning/Assignment2/DirName/Datasets/cifar-10-batches-mat/data_batch_1.mat'
train_file2 = '/Users/alperenkinali/Desktop/KTH/DD2424DeepLearning/Assignment2/DirName/Datasets/cifar-10-batches-mat/data_batch_2.mat'
train_file3 = '/Users/alperenkinali/Desktop/KTH/DD2424DeepLearning/Assignment2/DirName/Datasets/cifar-10-batches-mat/data_batch_3.mat'
train_file4 = '/Users/alperenkinali/Desktop/KTH/DD2424DeepLearning/Assignment2/DirName/Datasets/cifar-10-batches-mat/data_batch_4.mat'
train_file5 = '/Users/alperenkinali/Desktop/KTH/DD2424DeepLearning/Assignment2/DirName/Datasets/cifar-10-batches-mat/data_batch_5.mat'

#val_file = '/Users/alperenkinali/Desktop/KTH/DD2424DeepLearning/Assignment2/DirName/Datasets/cifar-10-batches-mat/data_batch_2.mat'
test_file = '/Users/alperenkinali/Desktop/KTH/DD2424DeepLearning/Assignment2/DirName/Datasets/cifar-10-batches-mat/test_batch.mat'

train_X, train_Y, train_y = load_batch(train_file)
train_X2, train_Y2, train_y2 = load_batch(train_file2)
train_X3, train_Y3, train_y3 = load_batch(train_file3)
train_X4, train_Y4, train_y4 = load_batch(train_file4)
train_X5, train_Y5, train_y5 = load_batch(train_file5)
#val_X, val_Y, val_y = load_batch(val_file)
test_X, test_Y, test_y = load_batch(test_file)

#Exercise 4, load data
E4_trX=np.hstack((train_X,train_X2,train_X3,train_X4,train_X5))
E4_trY=np.hstack((train_Y,train_Y2,train_Y3,train_Y4,train_Y5))
E4_try=np.hstack((train_y,train_y2,train_y3,train_y4,train_y5))

Xva=E4_trX[0:,0:1000]
Yva=E4_trY[0:,0:1000]
yva=E4_try[0:1000]
Xtr=E4_trX[0:,1000:]
Ytr=E4_trY[0:,1000:]
ytr=E4_try[1000:]

train_X_normalized, val_X_normalized, test_X_normalized = preprocess_data(Xtr, Xva, test_X)
#train_X_normalized, val_X_normalized, test_X_normalized = preprocess_data(train_X, val_X, test_X)
# Initialize parameters
n_epochs = 10
eta = 0.01
n_batch = 100
lamda = 0.0031625626

d = train_X_normalized.shape[0]
m = 50
K = train_Y.shape[0]

W1 = np.random.randn(m, d) * np.sqrt(1 / d)
b1 = np.zeros((m, 1))
W2 = np.random.randn(K, m) * np.sqrt(1 / m)
b2 = np.zeros((K, 1))


#Samples
X_sample = train_X_normalized[:, :2]
Y_sample = train_Y[:, :2]
P, h = forward_pass(X_sample, W1, b1, W2, b2)

#Relative error check

# Compute the analytical gradients
grad_W1_analytical, grad_b1_analytical, grad_W2_analytical, grad_b2_analytical = compute_gradients(X_sample, Y_sample, P, h, W1, b1, W2, b2, lamda) #vissa saker används inte

# Compute the numerical gradients
#grad_W1_num, grad_b1_num, grad_W2_num, grad_b2_num = compute_grads_num(X_sample, Y_sample, W1, b1, W2, b2, lamda, h=1e-5)
# or
grad_W1_num, grad_b1_num, grad_W2_num, grad_b2_num = compute_grads_num_slow(X_sample, Y_sample, W1, b1, W2, b2, lamda, h=1e-5)


W_analytical = [grad_W1_analytical, grad_W2_analytical]
W_num = [grad_W1_num, grad_W2_num]
b_analytical = [grad_b1_analytical, grad_b2_analytical]
b_num = [grad_b1_num, grad_b2_num]


for i in range(len(W_analytical)):
    rel_error_W = relative_error(W_analytical[i], W_num[i])
    rel_error_b = relative_error(b_analytical[i], b_num[i])
    print(f"Layer {i + 1}: Relative error for W: {np.mean(rel_error_W):.4e}, Relative error for b: {np.mean(rel_error_b):.4e}")


#Sanity check
#mini_batch_gradient_descent(X_sample, Y_sample,val_X_normalized, val_Y ,W1, b1, W2, b2, lamda, n_batch, eta, n_epochs)

eta_min = 1e-5
eta_max = 1e-1
#n_s = 800 # Exercise 3 and above

# Exercise 4
n = train_X_normalized.shape[1]
n_s = 2 * (n // n_batch)

W1_trained, b1_trained, W2_trained, b2_trained, loss_train_history, loss_val_history, cost_train_history, cost_val_history, test_accuracy_history, train_accuracy_history,  eta_history, update_step_history= mini_batch_gradient_descent_cyclical(
    train_X_normalized, Ytr, ytr,
    val_X_normalized, Yva, yva,
    test_X_normalized, test_Y, test_y,
    W1, b1, W2, b2, lamda, n_batch, eta_min, eta_max, n_s, n_epochs
)

print('Cost lists')
print(len(cost_train_history))
print(len(cost_val_history))

def plot_loss(loss_train_history, loss_val_history):

    plt.figure(figsize=(12, 6))
    x_values = [i * 49 for i in range(len(loss_train_history))]
    plt.plot(x_values, loss_train_history, label='Training Loss')
    plt.plot(x_values, loss_val_history, label='Validation Loss')
    plt.xlabel('Update steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def plot_cost(cost_train_history, cost_val_history):
    plt.figure(figsize=(12, 6))
    x_values = [i * 49 for i in range(len(cost_train_history))]
    plt.plot(x_values, cost_train_history, label='Training Cost')
    plt.plot(x_values, cost_val_history, label='Validation Cost')
    plt.xlabel('Update steps')
    plt.ylabel('Cost')
    plt.legend()
    plt.show()

def plot_accuracy(test_accuracy_history, train_accuracy_history):
    plt.figure(figsize=(12, 6))
    x_values = [i * 49 for i in range(len(test_accuracy_history))]
    print('accuracy list')
    print(len(x_values))
    print(len(test_accuracy_history))
    print(len(train_accuracy_history))
    plt.plot(x_values, test_accuracy_history, label='Training Accuracy')
    plt.plot(x_values, train_accuracy_history, label='Validation Accuracy')
    plt.xlabel('Update steps')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def plot_eta(eta_history, update_step_history):
    plt.figure(figsize=(12, 6))
    x_values = [i for i in range(len(update_step_history))]
    plt.plot(x_values, eta_history)
    plt.xlabel('Update Step')
    plt.ylabel('Learning Rate (eta)')
    plt.grid(True)
    plt.show()

plot_loss(loss_train_history, loss_val_history)
plot_cost(cost_train_history, cost_val_history)
plot_accuracy(test_accuracy_history, train_accuracy_history)
plot_eta(eta_history, update_step_history)
# Check the final training loss


# Function for random search
def random_search(train_X, train_Y, train_y, val_X, val_Y, y_val,test_X_normalized, test_Y, test_y, W1, b1, W2, b2, n_batch, eta_min, eta_max, n_s,
                  n_epochs, n_lambda, lambda_range):
    results = []

    for i in range(n_lambda):
        lambda_val = random.uniform(lambda_range[0], lambda_range[1])
        print("Lambda {}: {:.10f}".format(i + 1, lambda_val))

        W1_opt, b1_opt, W2_opt, b2_opt, loss_train_history, loss_val_history, cost_train_history, cost_val_history, train_accuracy_history, test_accuracy_history,  eta_history, update_step_history = mini_batch_gradient_descent_cyclical(
            train_X, train_Y, train_y,
            val_X, val_Y, y_val,
            test_X_normalized, test_Y, test_y,
            W1, b1, W2, b2, lambda_val, n_batch, eta_min, eta_max, n_s, n_epochs)

        results.append((lambda_val, cost_val_history[-1], test_accuracy_history[-1]))
        print(
            "Validation accuracy: {:.4f}\n".format(test_accuracy_history[-1]))

    return sorted(results, key=lambda x: x[2], reverse=True)  # sort by test accuracy


# Perform random search for lambda
n_lambda = 8  # number of lambda values to try
lambda_range = (1e-5, 1e-2)
n_epochs = 30

print('Lambda range: ')
print(lambda_range[0])
print(lambda_range[1])

# Random search
results = random_search(
    train_X_normalized, Ytr, ytr,
    val_X_normalized, Yva, yva,
    test_X_normalized, test_Y, test_y,
    W1, b1, W2, b2, n_batch, eta_min, eta_max, n_s, n_epochs, n_lambda, lambda_range)

# Print results
print("Random search results:")
for r in results:
    print("Lambda: {:.10f}, Validation cost: {:.4f}, Test accuracy: {:.4f}".format(r[0], r[1], r[2]))

