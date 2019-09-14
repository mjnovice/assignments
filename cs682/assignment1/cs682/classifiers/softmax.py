import numpy as np
from random import shuffle
from math import exp, log
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] # N
  num_classes = W.shape[1]
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_exp_score = exp(scores[y[i]])
    sum_exp_scores = 0
    for sc in scores:
        sum_exp_scores += exp(sc)
    loss += -log(correct_class_exp_score/sum_exp_scores)
    for j in range(num_classes):
      dW[:,j] += X[i]*((exp(scores[j])/sum_exp_scores) - correct_class_exp_score)
  loss /= num_train
  dW /= num_train
  loss += reg * np.sum(W * W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0] # N
  pscores = X.dot(W) # N x C
  pscores_exp = np.exp(pscores) # N x C
  correct_class_pscores_idx = (np.arange(0,num_train,1),(y))
  correct_class_pscores = pscores_exp[correct_class_pscores_idx] # N
  pscores_exp_rowwise_sum = np.sum(pscores_exp, axis=1)

  inner_exp=correct_class_pscores/pscores_exp_rowwise_sum
  loss = np.sum(-np.log(inner_exp))
  loss/=num_train
  loss += reg * np.sum(W * W)

  pscores_exp_rowwise_sum = pscores_exp_rowwise_sum.reshape(num_train,1) # N
  pscores_exp_norma = pscores_exp/pscores_exp_rowwise_sum # N x C
  pscores_exp_norma_minus_correct = pscores_exp_norma - correct_class_pscores.reshape(num_train,1)

  dW = X.T.dot(pscores_exp_norma_minus_correct)
  dW/=num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

