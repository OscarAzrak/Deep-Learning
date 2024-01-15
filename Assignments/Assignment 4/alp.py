import numpy as np
import warnings
import matplotlib.pyplot as plt
import copy


warnings.filterwarnings("ignore", category=RuntimeWarning)


# 0.1 Read in the data
book_fname = '/Users/alperenkinali/Desktop/KTH/DD2424DeepLearning/Assignment4/goblet_book.txt'
with open(book_fname, 'r') as f:
    book_data = f.read()

# Find the unique characters
book_chars = sorted(set(book_data))
K = len(book_chars)  # The dimensionality of the output/input vector of your RNN

# Create dictionaries to map characters to indices and vice versa
char_to_ind = {char: ind for ind, char in enumerate(book_chars)}
ind_to_char = {ind: char for ind, char in enumerate(book_chars)}

# 0.2 Set hyper-parameters & initialize the RNN’s parameters

# Set hyper-parameters
m = 100  # Dimensionality of the hidden state
eta = 0.1  # Learning rate
seq_length = 25  # Length of the input sequences during training

# Initialize the parameters of the RNN
class RNN:
    pass

rnn = RNN()

# Initialize the bias vectors to zero
rnn.b = np.zeros((m, 1))
rnn.c = np.zeros((K, 1))

# Initialize the weight matrices randomly
#sig = 0.01  # standard deviation for the normal distribution
#rnn.U = np.random.randn(m, K) * sig
#rnn.W = np.random.randn(m, m) * sig
#rnn.V = np.random.randn(K, m) * sig

sqrt1_over_n = lambda n: 1. / np.sqrt(n)
rnn.U = np.random.randn(m, K) * sqrt1_over_n(K)
rnn.W = np.random.randn(m, m) * sqrt1_over_n(m)
rnn.V = np.random.randn(K, m) * sqrt1_over_n(m)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def synthesize(rnn, h0, x0, n):
    # Initialize the hidden state
    h = np.copy(h0)

    # Initialize the first dummy input vector
    x = np.copy(x0)

    # Prepare to store the generated sequence
    Y = np.zeros((K, n))

    for t in range(n):
        # Compute next hidden state (equation 1)
        h = np.tanh(np.dot(rnn.W, h) + np.dot(rnn.U, x) + rnn.b)

        # Compute output (unnormalized probabilities) (equation 2)
        o = np.dot(rnn.V, h) + rnn.c

        # Compute normalized probabilities (equation 3)
        p = softmax(o)

        # Sample the next character (equation 4)
        x_next = np.random.choice(range(K), p=p.ravel())

        # Update the input to the next character
        x = np.zeros((K, 1))
        x[x_next] = 1

        # Store the generated character
        Y[:, t] = x.ravel()

    return Y

# Initialize h0 and x0
h0 = np.zeros((m, 1))
x0 = np.zeros((K, 1))
x0[char_to_ind['.']] = 1

# Synthesize a sequence of length 200
n = 1000
Y = synthesize(rnn, h0, x0, n)


# Convert the one-hot encoded sequence back into characters
text = ''.join(ind_to_char[np.argmax(Y[:, i])] for i in range(n))

print("Generated text: ", text)

def compute_loss_and_gradients(rnn, X, Y, h0):
    # Initialize loss and gradients
    loss = 0
    grads = {'W': np.zeros_like(rnn.W),
             'U': np.zeros_like(rnn.U),
             'V': np.zeros_like(rnn.V),
             'b': np.zeros_like(rnn.b),
             'c': np.zeros_like(rnn.c)}

    # Initialize the hidden state and output vectors for each time step
    H = np.zeros((m, X.shape[1] + 1))
    H[:, -1] = h0.ravel()
    O = np.zeros((K, X.shape[1]))

    # Forward pass
    for t in range(X.shape[1]):
        # Compute next hidden state (equation 1)
        H[:, t] = np.tanh(np.dot(rnn.W, H[:, t - 1]) + np.dot(rnn.U, X[:, t]) + rnn.b.ravel())

        # Compute output (unnormalized probabilities) (equation 2)
        O[:, t] = np.dot(rnn.V, H[:, t]) + rnn.c.ravel()

    # Compute softmax of O
    P = softmax(O)

    for t in range(X.shape[1]):
        # Compute loss (cross-entropy loss)
        loss += -np.log(P[Y[:, t] == 1, t])

    # Backward pass
    dH_next = np.zeros_like(H[:, 0])
    for t in reversed(range(X.shape[1])):
        # Compute gradient of loss w.r.t. output (equation for dL/do)
        dO = np.copy(P[:, t])
        dO[Y[:, t] == 1] -= 1

        # Compute gradients w.r.t. parameters (equations for dL/dV, dL/dc)
        grads['V'] += np.outer(dO, H[:, t])
        grads['c'] += dO.reshape(-1, 1)

        # Compute gradient w.r.t. hidden state (equation for dL/dh)
        dH = np.dot(rnn.V.T, dO) + dH_next
        dH_raw = (1 - H[:, t]**2) * dH

        # Compute gradients w.r.t. parameters (equations for dL/dW, dL/dU, dL/db)
        grads['W'] += np.outer(dH_raw, H[:, t - 1])
        grads['U'] += np.outer(dH_raw, X[:, t])
        grads['b'] += dH_raw.reshape(-1, 1)

        # Pass gradient back to next time step
        dH_next = np.dot(rnn.W.T, dH_raw)

    # Gradient clipping
    for f in grads:
        grads[f] = np.clip(grads[f], -5, 5)

    return loss, grads

def compute_grad_num(X, Y, f, RNN, h):
    n = np.prod(RNN.__dict__[f].shape)
    grad = np.zeros_like(RNN.__dict__[f])
    hprev = np.zeros((RNN.W.shape[0], 1))

    for i in range(n):
        RNN_try = copy.deepcopy(RNN)
        RNN_try.__dict__[f].flat[i] -= h
        l1, _ = compute_loss_and_gradients(RNN_try, X, Y, hprev)

        RNN_try = copy.deepcopy(RNN)
        RNN_try.__dict__[f].flat[i] += h
        l2, _ = compute_loss_and_gradients(RNN_try, X, Y, hprev)

        grad.flat[i] = (l2 - l1) / (2 * h)

    return grad

def compute_grads_num(X, Y, RNN, h):
    num_grads = {}
    for f in RNN.__dict__.keys():
        print('Computing numerical gradient for', 'Field name:', f)
        num_grads[f] = compute_grad_num(X, Y, f, RNN, h)
    return num_grads

# Convert the string data into integer representation
book_data_indices = [char_to_ind[char] for char in book_data]

# Convert the integer representation into one-hot encoding
X_all = np.zeros((K, len(book_data)))
X_all[book_data_indices, np.arange(len(book_data))] = 1

# Create input and target sequences
X = X_all[:, :-1]  # all characters except the last one
Y = X_all[:, 1:]  # all characters except the first one

# Initialize h
h = 1e-5

#Gradient check
# Sequence lengths to check
seq_lengths = [10, 20, 30]

# For each sequence length
for seq_length in seq_lengths:
    print(f"Sequence length: {seq_length}")

    # Generate sequences
    X_seq = X[:, :seq_length]
    Y_seq = Y[:, :seq_length]

    # Compute numerical gradients
    num_grads = compute_grads_num(X_seq, Y_seq, rnn, h)

    # Compute analytical gradients
    _, anal_grads = compute_loss_and_gradients(rnn, X_seq, Y_seq, h0)

    # Compute and print relative errors
    rel_errors = {}
    for f in rnn.__dict__.keys():
        num_grad = num_grads[f]
        anal_grad = anal_grads[f]

        numerator = np.abs(num_grad - anal_grad)
        denominator = np.maximum(1e-6, np.abs(num_grad) + np.abs(anal_grad))
        rel_error = numerator / denominator

        rel_errors[f] = rel_error

    for f, rel_error in rel_errors.items():
        print(f"Relative error for {f}: {np.mean(rel_error)}")
    print("\n")


# Initialize the AdaGrad memory vectors
rnn.mW = np.zeros_like(rnn.W)
rnn.mU = np.zeros_like(rnn.U)
rnn.mV = np.zeros_like(rnn.V)
rnn.mb = np.zeros_like(rnn.b)
rnn.mc = np.zeros_like(rnn.c)

# Initialize some other variables
e = 0
smooth_loss = -np.log(1.0 / K) * seq_length  # Initial loss
smooth_losses = []  # List to store smooth loss values

# Training loop
for i in range(300100):
    # Prepare inputs and targets
    if e + seq_length + 1 >= len(book_data) or e == 0:
        hprev = np.zeros((m, 1))
        e = 0
    inputs = [char_to_ind[ch] for ch in book_data[e:e+seq_length]]
    targets = [char_to_ind[ch] for ch in book_data[e+1:e+seq_length+1]]

    # Forward pass
    X = np.zeros((K, seq_length))
    Y = np.zeros((K, seq_length))
    for j in range(seq_length):
        X[inputs[j], j] = 1
        Y[targets[j], j] = 1
    loss, grads = compute_loss_and_gradients(rnn, X, Y, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # Print the smooth loss every 100 steps
    if i % 100 == 0:
        print(f"Iteration {i}, smooth loss: {smooth_loss}")
        smooth_losses.append(smooth_loss)

    # Backward pass and AdaGrad update
    for param, grad, mem in zip(['W', 'U', 'V', 'b', 'c'],
                                [grads['W'], grads['U'], grads['V'], grads['b'], grads['c']],
                                ['mW', 'mU', 'mV', 'mb', 'mc']):
        rnn.__dict__[mem] += grads[param] * grads[param]
        rnn.__dict__[param] += -eta * grads[param] / np.sqrt(rnn.__dict__[mem] + 1e-8)

    # Synthesize text every 500 steps
    if i % 10000 == 0:
        print("\nSynthesized text:")
        x0 = np.zeros((K, 1))
        x0[inputs[0]] = 1
        Y_synth = synthesize(rnn, hprev, x0, 1000)
        text_synth = ''.join(ind_to_char[np.argmax(Y_synth[:, i])] for i in range(1000))
        print(text_synth)

    e += seq_length

plt.figure(figsize=(12, 6))
plt.plot(smooth_losses)
plt.xlabel('Iteration (x100)')
plt.ylabel('Smooth Loss')
plt.title('Smooth Loss Over Time')
plt.show()