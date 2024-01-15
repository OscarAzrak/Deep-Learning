import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import functions as f
from sklearn.preprocessing import OneHotEncoder
import math

np.random.seed(1337)
enc = OneHotEncoder(sparse_output=False)

"""
Assignment 1:
"""

# Exercise 1
# Read the data


def load_data():
    train_test_val = {
        "train": "data_batch_1",
        "test": "test_batch",
        "val": "data_batch_2",
    }

    for key, file in train_test_val.items():
        dataset = f.LoadBatch(file)

        n = len(dataset[b"data"])
        X = np.array(dataset[b"data"]).T
        Y = enc.fit_transform(np.array(dataset[b"labels"]).reshape(-1, 1)).T
        y = np.array(dataset[b"labels"])

        train_test_val[key] = [X, Y, y]

    pre_process_data(train_test_val)

    return train_test_val


def load_all_data():
    train_test_val = {
        "train": ["data_batch_1", "data_batch_3", "data_batch_4", "data_batch_5"],
        "test": "test_batch",
        "val": "data_batch_2",
    }

    for key, file in train_test_val.items():
        # check if the data type is a list
        X, Y, y = [], [], []
        if isinstance(file, list):
            for fi in file:
                dataset = f.LoadBatch(fi)
                n = len(dataset[b"data"])
                X.append(np.array(dataset[b"data"]).T)
                Y.append(
                    enc.fit_transform(np.array(dataset[b"labels"]).reshape(-1, 1)).T
                )
                y.append(np.array(dataset[b"labels"]))
            X = np.concatenate(X, axis=1)
            Y = np.concatenate(Y, axis=1)
            y = np.concatenate(y, axis=0)

        else:
            dataset = f.LoadBatch(file)

            n = len(dataset[b"data"])
            X = np.array(dataset[b"data"]).T
            Y = enc.fit_transform(np.array(dataset[b"labels"]).reshape(-1, 1)).T
            y = np.array(dataset[b"labels"])

        print(X.shape, Y.shape, y.shape)

        train_test_val[key] = [X, Y, y]

    pre_process_data(train_test_val)

    return train_test_val


def pre_process_data(train_test_val):
    # Pre-process the data
    mean_X = np.mean(train_test_val["train"][0], axis=1).reshape(-1, 1)
    std_X = np.std(train_test_val["train"][0], axis=1).reshape(-1, 1)

    for key in train_test_val.keys():
        train_test_val[key][0] = (train_test_val[key][0] - mean_X) / std_X

    return train_test_val


# Exercise 2
# Compute the gradients for the network parameters


def evaluate_classifier(X, w1, w2, b1, b2):
    s1 = w1 @ X + b1
    h = f.ReLU(s1)
    s = w2 @ h + b2
    p = f.softmax(s)
    return p, h


def compute_cost(X, Y, w1, w2, b1, b2, lamda):
    p, _ = evaluate_classifier(X, w1, w2, b1, b2)
    N = X.shape[1]
    loss = -np.sum(Y * np.log(p)) / N
    reg = lamda * (np.sum(w1**2) + np.sum(w2**2)) / 2
    return loss + reg


def compute_accuracy(X, y, w1, w2, b1, b2):
    p, _ = evaluate_classifier(X, w1, w2, b1, b2)
    y_pred = np.argmax(p, axis=0)
    return np.sum(y_pred == y) / len(y)


def compute_gradients(X, Y, w1, w2, b1, b2, lamda):
    N = X.shape[1]
    p, h = evaluate_classifier(X, w1, w2, b1, b2)

    g = -(Y - p)
    grad_w2 = np.matmul(g, h.T) / N + 2 * lamda * w2
    grad_b2 = np.sum(g, axis=1).reshape(-1, 1) / N

    g = np.matmul(w2.T, g)
    g[h <= 0.0] = 0.0
    grad_w1 = np.matmul(g, X.T) / N + 2 * lamda * w1
    grad_b1 = np.sum(g, axis=1).reshape(-1, 1) / N

    return grad_w1, grad_w2, grad_b1, grad_b2


def compute_gradients_num(X, Y, w1, w2, b1, b2, lamda, h=1e-6):
    grad_w1 = np.zeros(w1.shape)
    grad_w2 = np.zeros(w2.shape)
    grad_b1 = np.zeros(b1.shape)
    grad_b2 = np.zeros(b2.shape)

    c = compute_cost(X, Y, w1, w2, b1, b2, lamda)

    for i in range(len(b1)):
        b1[i] += h
        c2 = compute_cost(X, Y, w1, w2, b1, b2, lamda)
        grad_b1[i] = (c2 - c) / h
        b1[i] -= h

    for i in range(len(b2)):
        b2[i] += h
        c2 = compute_cost(X, Y, w1, w2, b1, b2, lamda)
        grad_b2[i] = (c2 - c) / h
        b2[i] -= h

    for i in range(len(w1)):
        for j in range(len(w1[0])):
            w1[i][j] += h
            c2 = compute_cost(X, Y, w1, w2, b1, b2, lamda)
            grad_w1[i][j] = (c2 - c) / h
            w1[i][j] -= h

    for i in range(len(w2)):
        for j in range(len(w2[0])):
            w2[i][j] += h
            c2 = compute_cost(X, Y, w1, w2, b1, b2, lamda)
            grad_w2[i][j] = (c2 - c) / h
            w2[i][j] -= h

    return grad_w1, grad_w2, grad_b1, grad_b2


def compare_gradients(X, Y, w1, w2, b1, b2, lamda):
    grad_w1, grad_w2, grad_b1, grad_b2 = compute_gradients(X, Y, w1, w2, b1, b2, lamda)
    grad_w1_num, grad_w2_num, grad_b1_num, grad_b2_num = compute_gradients_num(
        X, Y, w1, w2, b1, b2, lamda
    )

    print("grad_w1: ", np.max(np.abs(grad_w1 - grad_w1_num)))
    print("grad_w2: ", np.max(np.abs(grad_w2 - grad_w2_num)))
    print("grad_b1: ", np.max(np.abs(grad_b1 - grad_b1_num)))
    print("grad_b2: ", np.max(np.abs(grad_b2 - grad_b2_num)))


def update(w1, w2, b1, b2, grad_w1, grad_w2, grad_b1, grad_b2, eta):
    w1 -= eta * grad_w1
    w2 -= eta * grad_w2
    b1 -= eta * grad_b1
    b2 -= eta * grad_b2
    return w1, w2, b1, b2


# Exercise 3
# Train using cyclical learning rate


def train(w1, w2, b1, b2, data, eta_min, eta_max, lamda, n_epochs):
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    X_train, Y_train, y_train = data["train"]
    X_val, Y_val, y_val = data["val"]
    batch_size = 100
    step_size = X_train.shape[1] / batch_size

    cyclic_learning_rate = f.cyclical_learning_rate(
        eta_min, eta_max, n_epochs, step_size
    )
    eta = eta_min

    while cyclic_learning_rate.cycle < 3:
        # Shuffle the data
        X_train, Y_train, y_train = f.shuffle_data(X_train, Y_train, y_train)

        # print(f"Epoch {epoch + 1}/{n_epochs}")
        for batch in range(int(step_size)):
            X_batch = X_train[:, batch * batch_size : (batch + 1) * batch_size]
            Y_batch = Y_train[:, batch * batch_size : (batch + 1) * batch_size]

            grad_w1, grad_w2, grad_b1, grad_b2 = compute_gradients(
                X_batch, Y_batch, w1, w2, b1, b2, lamda
            )
            w1, w2, b1, b2 = update(
                w1, w2, b1, b2, grad_w1, grad_w2, grad_b1, grad_b2, eta
            )

            eta = cyclic_learning_rate()

        train_loss.append(compute_cost(X_train, Y_train, w1, w2, b1, b2, lamda))
        train_acc.append(compute_accuracy(X_train, y_train, w1, w2, b1, b2))
        val_loss.append(compute_cost(X_val, Y_val, w1, w2, b1, b2, lamda))
        val_acc.append(compute_accuracy(X_val, y_val, w1, w2, b1, b2))

        # print(f"Learning rate: {eta:.5f}")

        print(f"Train loss: {train_loss[-1]:.3f}")
        print(f"Train acc: {train_acc[-1]:.3f}")
        print(f"Val loss: {val_loss[-1]:.3f}")
        print(f"Val acc: {val_acc[-1]:.3f}")

    return (
        w1,
        w2,
        b1,
        b2,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
    )


def plot_loss_acc(train_loss, train_acc, val_loss, val_acc):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train")
    plt.plot(val_loss, label="val")
    plt.title("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="train")
    plt.plot(val_acc, label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.savefig("loss_acc.png")
    plt.show()


if __name__ == "__main__":
    w1 = np.random.normal(0, 0.01, (50, 3072))
    w2 = np.random.normal(0, 0.01, (10, 50))
    b1 = np.zeros((50, 1))
    b2 = np.zeros((10, 1))

    train_test_val = load_all_data()

    # Compare gradients
    # X, Y, y = train_test_val["train"]
    # X = X[:, :5]
    # Y = Y[:, :5]
    # lamda = 0.01
    # compare_gradients(X, Y, w1, w2, b1, b2, lamda)

    eta_min = 1e-5
    eta_max = 1e-1
    n_epochs = 500

    l_low = -5
    l_high = -1
    lamdas = f.grid_search_lamda(l_low, l_high)
    best_lamda = 0
    best_acc = float("-inf")
    # Create a dict for a lambda and the corresponding accuracy
    lamda_acc = {}
    lamda_acc[0] = float("-inf")

    for lamda in lamdas:
        print(f"Grid Search Lambda: {lamda}")
        (
            final_w1,
            final_w2,
            final_b1,
            final_b2,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        ) = train(w1, w2, b1, b2, train_test_val, eta_min, eta_max, lamda, n_epochs)

        lamda_acc[lamda] = np.max(val_acc)

    best_lamda = max(lamda_acc, key=lamda_acc.get)

    print(f"Best Lambda: {best_lamda}")
    best_lamda_acc = {}
    best_lamda_acc[0] = float("-inf")

    fine_lamdas = f.fine_search_lamda(best_lamda)
    for lamda in fine_lamdas:
        print(f"Fine Search Lambda: {lamda}")
        (
            final_w1,
            final_w2,
            final_b1,
            final_b2,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        ) = train(w1, w2, b1, b2, train_test_val, eta_min, eta_max, lamda, n_epochs)

        best_lamda_acc[lamda] = np.max(val_acc)

    best_best_lamda = max(best_lamda_acc, key=best_lamda_acc.get)

    # final traning with best best lambda
    print(f"Best Best Lambda: {best_best_lamda}")
    (
        final_w1,
        final_w2,
        final_b1,
        final_b2,
        train_loss,
        train_acc,
        val_loss,
        val_acc,
    ) = train(
        w1, w2, b1, b2, train_test_val, eta_min, eta_max, best_best_lamda, n_epochs
    )

    # Test the model
    X_test, Y_test, y_test = train_test_val["test"]
    test_loss = compute_cost(
        X_test, Y_test, final_w1, final_w2, final_b1, final_b2, lamda
    )
    test_acc = compute_accuracy(X_test, y_test, final_w1, final_w2, final_b1, final_b2)
    print(f"Test loss: {test_loss:.3f}")
    print(f"Test acc: {test_acc:.3f}")

    # Print the top 3 lambda with the highest accuracy
    print("Top 3 Lambda with the highest accuracy")
    for i in range(3):
        best_lamda = max(lamda_acc, key=lamda_acc.get)
        print(f"Lambda: {best_lamda} Accuracy: {lamda_acc[best_lamda]}")
        del lamda_acc[best_lamda]

    print("Top 3 fine Lambda with the highest accuracy")
    for i in range(3):
        best_lamda = max(best_lamda_acc, key=best_lamda_acc.get)
        print(f"Lambda: {best_lamda} Accuracy: {best_lamda_acc[best_lamda]}")
        del best_lamda_acc[best_lamda]

    plot_loss_acc(train_loss, train_acc, val_loss, val_acc)
