from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax
    32x32x32 - 32x32x32 - 16x16x32 - 8192 - 8192 - 8192x1 - 8192x10' - 10x1

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        std = weight_scale
        output_size = num_classes
        C,H,W = input_dim
        
        self.params['W1'] = np.random.normal(0, weight_scale, (num_filters, C, filter_size, filter_size))
        self.params['b1'] = np.zeros(num_filters)
        self.params['W2'] = np.random.normal(0, weight_scale, (num_filters*W/2*H/2, hidden_dim))
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = np.random.normal(0, std, (hidden_dim, output_size))
        self.params['b3'] = np.zeros(output_size)
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        
        #Convolution + Relu Layer not used because for first convolution we need to convert to im2col
        
        #Convolution Layer
        conv_out, conv_cache = conv_forward_im2col(X, W1, b1, conv_param)
        
        #Relu Layer
        relu_out, relu_cache = relu_forward(conv_out)
        
        #Maxpool Layer
        max_pool_out, max_pool_cache = max_pool_forward_fast(relu_out, pool_param)
        
        #Affine + Relu Layer
        affine_relu_out, affine_relu_cache = affine_relu_forward(max_pool_out, W2, b2)
        
        #Affine Layer
        affine2_out, affine2_cache = affine_forward(affine_relu_out, W3, b3)
        scores = affine2_out
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        #conv - relu - 2x2 max pool - affine - relu - affine - softmax#
        
        #Calculating loss
        data_loss, dscores = softmax_loss(scores,y)
        reg_loss = 0.5*self.reg*np.sum(self.params['W1']*self.params['W1']) + 0.5*self.reg*np.sum(self.params['W2']*self.params['W2']) + 0.5*self.reg*np.sum(self.params['W3']*self.params['W3'])
        loss = data_loss + reg_loss
        
        #Calculating Gradients and Back-prop
        # Backprop in affine layer
        affine2_dx, affine2_dw, affine2_db = affine_backward(dscores, affine2_cache)
        grads['W3'] = affine2_dw + self.reg * W3
        grads['b3'] = affine2_db
         
        affine1_dx, affine1_dw, affine1_db = affine_relu_backward(affine2_dx, affine_relu_cache)
        grads['W2'] = affine1_dw + self.reg * W2
        grads['b2'] = affine1_db
    
        max_pool_dx = max_pool_backward_fast(affine1_dx, max_pool_cache)
        relu_dx = relu_backward(max_pool_dx, relu_cache)
        conv_dx, conv_dw, conv_db = conv_backward_im2col(relu_dx, conv_cache)
        grads['W1'] = conv_dw + self.reg * W1
        grads['b1'] = conv_db

        
        
        
        #pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
