import numpy as np
from random import shuffle


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
  stable_scores = []

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    # compute the loss and the gradient
    
    #############################################################################
  # f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
  # p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup
  # instead: first shift the values of f so that the highest number is 0:
  # f -= np.max(f) # f becomes [-666, -333, 0]
  # p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer                                                          #
  #############################################################################
  num_classes = W.shape[1]
  num_train = X.shape[0]
  for i in range(num_train):
    scores = X[i].dot(W)
    loss_i = -scores[y[i]] + np.log(sum(np.exp(scores)))
    loss += loss_i
    softmax_output = np.exp(scores)/np.sum(np.exp(scores))
    for j in range(num_classes):
        if y[i] == j:
            dW[:,j] += -X[i] * (1-softmax_output[j])
        else:
             dW[:,j] += -X[i] * (-softmax_output[j])
#    for j in range(num_classes):
#      if j == y[i]:
#        continue
#        loss += -np.log(np.exp(stable_scores)/(np.sum(np.exp(stable_scores))))
        #loss += -np.log(np.exp(correct_class_score + correct_class_score_stable) / float(np.sum(np.exp(scores+correct_class_score_stable)))
    #dW[:, i] += (p(i) - (i == y[i])) * X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  #dW /= num_train
  # regularize the weights
  # Add regularization to the loss.
  loss /= num_train 
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train 
  dW += reg*W
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
  num_train = X.shape[0]
  scores = np.dot(X,W)
  #scores -= np.max(scores)
  #correct_scores = scores[np.arange(num_train), y]
  c_loss = -scores[:,y] + np.log(np.sum(np.exp(scores),axis=1))
  loss = np.mean(c_loss)
  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  
  p_scores = np.exp(scores)/np.sum(np.exp(scores))
  dscores = p_scores
  dscores[range(num_train),y] -= 1
  dscores /= num_train
  dW = np.dot(X.T, dscores)
  #dW /= num_train
  dW += reg*W
  # Regularize
    
  #pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

