import numpy as np
import scipy.special
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!

# Q 2.1.2
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    #var_W = 2 / (in_size + out_size)
    var_W = np.sqrt(6/(in_size + out_size))
    W = np.random.uniform(-var_W, var_W, size=(in_size, out_size))
    b = np.zeros(out_size)
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    x[x < -300] = -300
    res = 1 / (1 + np.exp(-x))

    eps = 0.0000000000000000000000000001
    res[np.abs(res) < eps] = 0
    return res

# Q 2.2.1
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    d, k = W.shape
    n = len(X)
    y = np.zeros((n, k))

    for i in range(n):
        y[i] = W.T.dot(X[i]) + b

    y = np.dot(X, W) + b
    pre_act = y
    post_act = activation(y)

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):

    def softmax_row(row):
        row = np.exp(row)
        tmp = np.sum(row)
        row /= tmp
        return row

    res = np.apply_along_axis(softmax_row, 1, x)

    return res


# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    loss = -np.sum(y * np.log(probs))

    def pick(row):
        hot = np.argmax(row)
        row = np.zeros(row.shape)
        row[hot] = 1
        return row

    pred = np.apply_along_axis(pick, 1, probs)
    acc = np.sum(pred * y) / len(y)

    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

# Q 2.3.1
def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X
    #                          J (n, k)
    grad_W = np.zeros(W.shape) # (d,k)
    grad_X = np.zeros(X.shape)
    grad_b = np.zeros(b.shape)
    d, k = W.shape
    n = len(X)

    delta = delta * activation_deriv(post_act)

    for i in range(n):
        grad_W += np.expand_dims(X[i], axis=1).dot(np.expand_dims(delta[i], axis=0))
        grad_b += delta[i]
        grad_X[i, :] = W.dot(np.expand_dims(delta[i], axis=1)).reshape([-1])

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4.1
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    batches = []

    assert len(x) == len(y)
    p = np.random.permutation(len(x))
    x = x[p]
    y = y[p]

    while len(x) >= batch_size:
        batches.append((x[:batch_size], y[:batch_size]))
        x = x[batch_size:]
        y = y[batch_size:]

    return batches

