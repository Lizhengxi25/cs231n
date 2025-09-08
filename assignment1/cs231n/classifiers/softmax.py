from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    N = X.shape[0] ; C = W.shape[1]
    SX = np.zeros_like(W)
    S2 = np.zeros_like(W)
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        s_p = p.sum()
        p /= s_p  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
 
        One = np.exp(scores)
        One = One.reshape(1, -1)

        dSumdW = (X[i].T).reshape(-1, 1).dot(One)
        dlndW = dSumdW / s_p

        One_2 = np.zeros((1, C))
        One_2[0][y[i]] = -1

        dLidW = (X[i].T).reshape(-1, 1).dot(One_2) + dlndW
        dW += dLidW


    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    
    dW = dW / N + 2 * reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################

    C = W.shape[1]
    N = X.shape[0]

    scores = X.dot(W)

    MAX_row = np.max(scores, axis=1)
    MAX_matrix = MAX_row.reshape((-1, 1)).dot(np.ones((1, C)))
    # print(f"MAX_matrix shape is %f", MAX_matrix.shape)
    scores -= MAX_matrix
    scores = np.exp(scores)
    scores = (scores.T / np.sum(scores, axis=1).T).T
    
    tmp = scores[np.arange(N), y]
    L = - np.log(tmp)
    loss = np.sum(L) / N + reg * np.sum(W * W)


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################

    scores[np.arange(N), y] -= 1
    dLidW = X.T.dot(scores)
    dW = dLidW / N + 2 * reg * W

    return loss, dW
