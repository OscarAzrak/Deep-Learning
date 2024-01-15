# Reading the text file
with open('data/goblet_book.txt', 'r') as file:
    book_data = file.read()

# Getting unique characters
book_chars = sorted(set(book_data))
K = len(book_chars)  # dimensionality of the output (input) vector of your RNN

# Initializing maps

char_to_ind = {char: ind for ind, char in enumerate(book_chars)}
ind_to_char = {ind: char for ind, char in enumerate(book_chars)}



# Now, char_to_ind and ind_to_char can be used for converting between characters and their corresponding indices


# test using text "hello"
hello = 'hello'
hello_ind = [char_to_ind[char] for char in hello]
print(hello_ind)
print([ind_to_char[ind] for ind in hello_ind])




import numpy as np

# Set hyper-parameters
m = 100  # dimensionality of the hidden state
eta = 0.1  # learning rate
seq_length = 25  # length of the input sequences

# Initialize the RNN's parameters


class RNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1, seq_length=25, sigma=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        

        # Initialize weights and biases
        self.W = np.random.randn(hidden_size, hidden_size) / np.sqrt(hidden_size)  # weights for hidden states
        self.U = np.random.randn(hidden_size, input_size) / np.sqrt(input_size)  # weights for inputs
        self.b = np.zeros((hidden_size, 1))                        # bias for hidden states
        self.V = np.random.randn(output_size, hidden_size) / np.sqrt(hidden_size) # weights for output
        self.c = np.zeros((output_size, 1))                        # bias for output


# Initialize the RNN
rnn = RNN(K, m, K, learning_rate=eta, seq_length=seq_length)


def synthesize(rnn, h0, x0, n):
    h = h0
    x = x0
    Y = np.zeros((rnn.output_size, n))
    print(h.shape)

    for t in range(n):
        # Compute the hidden state
        a = np.dot(rnn.W, h.ravel()) + np.dot(rnn.U, x.ravel()) + rnn.b.ravel()
        h = np.tanh(a)

        # Compute the output
        o = np.dot(rnn.V, h) + rnn.c.ravel()
        p = np.exp(o) / np.sum(np.exp(o))  # normalize to get probabilities

        # Sample a character index from the probability distribution
        ix = np.random.choice(range(rnn.output_size), p=p.ravel())

        # Update the input for the next time step
        x = np.zeros((rnn.input_size, 1))
        x[ix] = 1

        # Store the one-hot representation of the sampled character
        Y[:, t] = x.ravel()


    return Y

def one_hot_seq_to_char_seq(one_hot_seq, ind_to_char):
    N = one_hot_seq.shape[1]
    char_seq = ''.join([ind_to_char[np.argmax(one_hot_seq[:, i])] for i in range(N)])
    return char_seq

h0 = np.zeros((m, 1))
x0 = np.zeros((K, 1))
n = 100

Y = synthesize(rnn, h0, x0, n)
#text = ''.join(ind_to_char[np.argmax(Y[:, i])] for i in range(n))
text = one_hot_seq_to_char_seq(Y, ind_to_char)
print(text)

import numpy as np
import copy

def compute_grad_num(X, Y, f, rnn, h):
    n = np.prod(getattr(rnn, f).shape)
    grad = np.zeros(getattr(rnn, f).shape)
    hprev = np.zeros((rnn.W.shape[0], 1))
    for i in range(n):
        RNN_try = copy.deepcopy(rnn)
        param = getattr(RNN_try, f).flatten()
        param[i] -= h
        setattr(RNN_try, f, param.reshape(getattr(rnn, f).shape))
        l1, _ = forward_backward_pass(RNN_try, X, Y, hprev)
        param[i] += 2*h
        setattr(RNN_try, f, param.reshape(getattr(rnn, f).shape))
        l2,_ = forward_backward_pass(RNN_try, X, Y, hprev)
        grad.flat[i] = (l2-l1) / (2*h)
    return grad

def compute_grads_num(X, Y, rnn, h):
    num_grads = {}
    for f in ['W', 'U', 'b', 'V', 'c']:
        print('Computing numerical gradient for')
        print('Field name: ', f)
        num_grads[f] = compute_grad_num(X, Y, f, rnn, h)
       
    return num_grads




def forward_backward_pass(rnn, X_chars, Y_chars, h0):

    # Initialize the hidden states and outputs
    h = np.zeros((rnn.b.shape[0], len(X_chars) + 1))
    h[:, 0] = h0.ravel()
    o = np.zeros((len(rnn.c), len(X_chars)))
    p = np.zeros((len(rnn.c), len(X_chars)))


    # Forward pass
    for t in range(X_chars.shape[1]):


        #a = np.dot(rnn.W, h.ravel()) + np.dot(rnn.U, x.ravel()) + rnn.b.ravel()
        #h = np.tanh(a)

        a1 = np.dot(rnn.W, h[:, t].ravel())
        a2 = np.dot(rnn.U, X_chars[:, t]) 
        a = a1 + a2 + rnn.b.ravel()
        h[:, t + 1] = np.tanh(a).ravel()
        o[:, t] = (np.dot(rnn.V, h[:, t + 1]) + rnn.c.ravel())

        p[:, t] = np.exp(o[:, t]) / np.sum(np.exp(o[:, t]))  # normalize to get probabilities

    loss = -np.sum(Y_chars * np.log(p[:, :Y_chars.shape[1]]))  # cross-entropy loss

    # Initialize the gradients
    grads = {
        'U': np.zeros_like(rnn.U),
        'W': np.zeros_like(rnn.W),
        'V': np.zeros_like(rnn.V),
        'b': np.zeros_like(rnn.b),
        'c': np.zeros_like(rnn.c)
    }

    # Backward pass
    dh_next = np.zeros_like(h[:, 0])
    for t in reversed(range(Y_chars.shape[1])):
        do = p[:, t] - Y_chars[:, t]
        grads['V'] += np.outer(do, h[:, t + 1])
        grads['c'] += do.reshape(rnn.c.shape)

        dh = np.dot(rnn.V.T, do) + dh_next
        da = (1 - h[:, t + 1]**2) * dh

        grads['U'] += np.outer(da, X_chars[:, t])
        grads['W'] += np.outer(da, h[:, t])
        grads['b'] += da.reshape(rnn.b.shape)

        dh_next = np.dot(rnn.W.T, da)


    # Clip the gradients to avoid exploding gradient problem
    for grad in grads.values():
        np.clip(grad, -5, 5, out=grad)

    return loss, grads
def relative_error(grad_num, grad_analytic):
    rel_errors = np.abs(grad_num - grad_analytic) / np.maximum(1e-6, np.abs(grad_num) + np.abs(grad_analytic))
    return np.mean(rel_errors)

X_chars = book_data[0:seq_length]
Y_chars = book_data[1:seq_length + 1]


def one_hot_encode(sequence, vocab_size):
    # Create an array of zeros with length of the sequence and vocab size
    one_hot = np.zeros((vocab_size, len(sequence)))

    # For each character in the sequence, change the corresponding index in the array to 1
    for i, char in enumerate(sequence):
        one_hot[char_to_ind[char], i] = 1.0  

    return one_hot

# One-hot encode the input sequences
X_chars_encoded = one_hot_encode(X_chars, K)
Y_chars_encoded = one_hot_encode(Y_chars, K)

#loss, (h, o, p) = forward_pass(rnn, X_chars_encoded, Y_chars_encoded, h0)
#grads = backward_pass(rnn, X_chars_encoded, Y_chars_encoded, h, p)


h = 1e-5

# Compute numerical gradients
num_grads = compute_grads_num(X_chars_encoded, Y_chars_encoded, rnn, h)

# Compute analytical gradients
_, ana_grads = forward_backward_pass(rnn, X_chars_encoded, Y_chars_encoded, h0)


# Compare gradients
for param in ['W', 'U', 'b', 'V', 'c']:
    rel_error = relative_error(num_grads[param], ana_grads[param])
    print(f'Relative Error for {param}: {rel_error}')


loss, grads = forward_backward_pass(rnn, X_chars_encoded, Y_chars_encoded, h0)


#loss, grads = forward_backward_pass(rnn, X_chars_encoded, Y_chars_encoded, h0)
for param, grad in grads.items():
    param_value = getattr(rnn, param)
    param_value -= eta * grad
    setattr(rnn, param, param_value)

print(loss)
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

def train_rnn(rnn, book_data, char_to_ind, ind_to_char, n_epochs):
    K = rnn.output_size
    seq_length = rnn.seq_length

    iter_per_epoch = len(book_data) // seq_length
    updates = n_epochs * iter_per_epoch
    smooth_loss = -np.log(1.0 / K) * seq_length  # loss at iteration 0
    hprev = np.zeros((rnn.hidden_size, 1))
    ada_params = {k: np.zeros_like(getattr(rnn, k)) for k in ['U', 'W', 'V', 'b', 'c']}

    smooth_loss_list = []
    best_loss = np.inf

    e= 0
    for update in tqdm(range(updates)):
        if e == 0 or e + seq_length + 1 > len(book_data):
            e = 1
            hprev = np.zeros((rnn.hidden_size, 1))  # reset RNN memory

        X_chars = book_data[e:e + seq_length]
        Y_chars = book_data[e + 1:e + seq_length + 1]

        X = one_hot_encode(X_chars, K)
        Y = one_hot_encode(Y_chars, K)

        loss, grads = forward_backward_pass(rnn, X, Y, hprev)

        # Update the parameters using Adagrad
        for param, grad in grads.items():
            param_value = getattr(rnn, param)
            ada_params[param] += grad**2
            param_value -= eta * grad / np.sqrt(ada_params[param] + 1e-8)
            setattr(rnn, param, param_value)

        smooth_loss = smooth_loss * 0.999 + loss * 0.001
        smooth_loss_list.append(smooth_loss)

        if smooth_loss < best_loss:
            rnn_best = copy.deepcopy(rnn)
            best_loss = smooth_loss

        if update % 10000 == 0:
            print("Smooth loss at step {}: {}".format(update, smooth_loss))
            Y_synthesized = synthesize(rnn, hprev, X[:, :1], 200)
            synthesized_seq = one_hot_seq_to_char_seq(Y_synthesized, ind_to_char)
            print("Update step {}: Synthesized text:\n".format(update), synthesized_seq)

        if update % 100000 == 0:
            Y_synthesized = synthesize(rnn, hprev, X[:, :1], 200)
            synthesized_seq = one_hot_seq_to_char_seq(Y_synthesized, ind_to_char)
            print("Synthesized text:\n", synthesized_seq)

    

        e += seq_length
    print(25*"_")

    Y_synthesized = synthesize(rnn_best, hprev, X[:, :1], 1000)
    synthesized_seq = one_hot_seq_to_char_seq(Y_synthesized, ind_to_char)
    print("Synthesized text:\n", synthesized_seq)
    plt.plot(smooth_loss_list)
    plt.title("Smooth loss over time")
    plt.xlabel("Iteration")
    plt.ylabel("Smooth loss")
    plt.show()
        

rnn = train_rnn(rnn, book_data, char_to_ind, ind_to_char, n_epochs=7)
