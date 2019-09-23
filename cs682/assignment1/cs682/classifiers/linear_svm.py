import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dW[:,j] += X[i,:]# X is NxD, while dw is DxC
        dW[:,y[i]] -= X[i,:]#j is the class being referred
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss and its gradient.
  loss += reg * np.sum(W * W)
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW

def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  prescores = X.dot(W)
  num_train = X.shape[0]
  correct_class_scores_idx = (np.arange(0,num_train,1),(y))
  correct_class_scores = prescores[correct_class_scores_idx]
  margins = prescores - correct_class_scores.reshape(num_train,1) + np.ones(prescores.shape)
  margins=margins.clip(min=0)
  margins[correct_class_scores_idx]=0
  loss=np.sum(margins)
  loss/=num_train
  loss += reg*np.sum(W*W)
  '''
  #the following is an array of dim N x C where N is the number in the training 
  #set and C is the number of classes
  allscores = X.dot(W)
  num_train = X.shape[0]
  correct_class_scores = np.zeros(num_train,dtype=float)
  correct_class_scores_idx = (np.arange(0,num_train,1),(y))
  correct_class_scores = allscores[correct_class_scores_idx]
  diffs = allscores - correct_class_scores.reshape(num_train,1)
  margins = diffs + np.ones(allscores.shape) #ones for the delta
  gtzeromask = margins > 0

  differences = np.sum(margins.clip(min=0))
  differences -= num_train #to compensate for the cases where j==yi
  loss = differences / num_train
  loss += reg * np.sum(W * W)
  '''
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask=margins>0
  #the normal case where j!=yi
  dW=X.T.dot(mask)

  masksum=np.sum(mask,axis=1)
  #axis expansion
  masksum_exp = masksum[...,np.newaxis]

  #Handling the case where j == yi or the correct classes
  defs=-masksum_exp*X

  #this allows repeated index addition and this is taking care of
  #the case of j == yi
  np.add.at(dW.T,y,defs)
  dW/=num_train
  #adding the reg loss gradient
  dW += 2*reg*W

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return loss, dW
