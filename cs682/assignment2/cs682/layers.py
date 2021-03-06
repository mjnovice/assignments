from builtins import range
import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    x_res = x.reshape(x.shape[0], np.prod(x.shape[1:]) )
    out = x_res.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    x_res = x.reshape(x.shape[0], np.prod(x.shape[1:]) )
    dx_res = dout.dot(w.T)
    dx = dx_res.reshape(x.shape)
    dw = x_res.T.dot(dout)
    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x.clip(min=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = dout*(x>0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))
    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################

        #calculating running_mean and running_var with the current batch
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)
        xc = (x - sample_mean)/np.sqrt(sample_var+ eps)
        out = xc*gamma + beta
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        xc = (x - running_mean)/np.sqrt(running_var+ eps)
        out = xc*gamma + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    cache = (x, gamma, beta, bn_param)
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    x, gamma, beta, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    n = x.shape[0]
    #let A = X - mean(x,axis=0)
    mu = np.mean(x, axis = 0) # D x 1
    a = x - mu # N x D
    varf = np.sqrt(np.var(x, axis=0)+ eps) #D x 1
    e = 1.0/varf #D x 1
    #backprop expanding E
    q = a*e #N x D
    dldq = dout*gamma
    dlda1 = dldq*e

    dlde = np.sum(a*dldq,axis=0) #D x 1
    #C is the matrix of the form 1/m * sum_of_all(xi-mean)^2 + eps
    dedc = -0.5 * e**3
    dcda2 = 2*a/n
    deda2 = dedc * dcda2#- e**3 * (a/n) # D x 1
    #dedc = (2./n) * a**3 #N x D
    #dqda2 = dcda2 * dqdc
    dlda2 = dlde * deda2

    dlda = dlda1 + dlda2

    dldgamma = np.sum(q * dout,axis=0)
    dldbeta = np.sum(dout, axis=0)

    dldmean = -1 * np.sum(dlda,axis=0)
    dmeandx = 1/n
    dldx = dldmean*dmeandx + dlda 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dldx, dldgamma, dldbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    x, gamma, beta, bn_param = cache
    eps = bn_param.get('eps', 1e-5)

    mu = np.mean(x, axis=0)
    a = x - mu # N x D
    varf = np.sqrt(np.var(x, axis=0)+ eps) #D x 1
    q = a/varf #N x D
    n=x.shape[0]
    b = 1+((mu*q)/(n*varf))
    c = (1 - (1./n))
    d= x/varf
    dfdq = gamma*dout
    dldx=(n*dfdq - np.sum(dfdq,axis=0)-q*np.sum(dfdq*q,axis=0))/(n*varf)
    dldgamma = np.sum(q * dout,axis=0)
    dldbeta = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dldx, dldgamma, dldbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """
    Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get('eps', 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    sample_mean = np.mean(x, axis=1,keepdims=True)
    sample_var = np.var(x, axis=1,keepdims=True)
    a=(x-sample_mean)
    xc = (a)/np.sqrt(sample_var+ eps)
    out = xc*gamma + beta
    cache = (x, gamma, beta, eps)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """
    Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    x, gamma, beta, eps = cache
    mu = np.mean(x, axis=1,keepdims=True)
    a = x - mu # N x D
    varf = np.sqrt(np.var(x, axis=1,keepdims=True)+ eps) #D x 1
    q = a/varf #N x D
    n=x.shape[1]
    dfdq = gamma*dout
    dldx=(n*dfdq - np.sum(dfdq,axis=1,keepdims=True)-q*np.sum(dfdq*q,axis=1,keepdims=True))/(n*varf)
    dldgamma = np.sum(q * dout,axis=0)
    dldbeta = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dldx, dldgamma, dldbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.

    NOTE: Please implement **inverted** dropout, not the vanilla version of dropout.
    See https://compsci682-fa18.github.io/notes/neural-networks-2/#reg for more details.

    NOTE 2: Keep in mind that p is the probability of **keep** a neuron
    output; this might be contrary to some sources, where it is referred to
    as the probability of dropping a neuron output.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None
    mask = (np.random.rand(x.shape[0],x.shape[1]) < p)/p
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        out = x*mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout*mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.


    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N,C,H,W = x.shape
    F,_,HH,WW = w.shape 
    pad = conv_param['pad']
    stride = conv_param['stride']
    Hprime = int(1 + (H + 2*pad - HH)/stride)
    Wprime = int(1 + (W + 2*pad - WW)/stride)
    out = np.zeros((N, F, Hprime, Wprime),dtype=float)
    npad=((0,0),(0,0),(pad,pad),(pad,pad))
    x_padded = np.pad(x,pad_width=npad,mode='constant',constant_values=0)
    for i in range(N):
        for f in range(F):
            for j in range(Hprime):
                for k in range(Wprime):
                    xj = j*stride 
                    xk = k*stride 
                    segmentA = x_padded[i,:,xj:xj+HH,xk:xk+WW]
                    segmentB = w[f]
                    out[i,f,j,k] = np.sum(segmentA*segmentB) + b[f]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    N = x.shape[0]
    C = x.shape[1]
    H = x.shape[2]
    W = x.shape[3]
    F = w.shape[0]
    C = w.shape[1]
    HH = w.shape[2]
    WW = w.shape[3]
    pad = conv_param['pad']
    stride = conv_param['stride']
    Hprime = int(1 + (H + 2*pad - HH)/stride)
    Wprime = int(1 + (W + 2*pad - WW)/stride)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    npad=((0,0),(0,0),(pad,pad),(pad,pad))
    x_padded = np.pad(x,pad_width=npad,mode='constant',constant_values=0)
    dx_padded = np.zeros_like(x_padded)
    for i in range(N):
        for f in range(F):
            for j in range(Hprime):
                for k in range(Wprime):
                    xj = j*stride 
                    xk = k*stride
                    segmentA = x_padded[i,:,xj:xj+HH,xk:xk+WW]
                    dx_padded[i,:,xj:xj+HH,xk:xk+WW] +=  w[f] * dout[i,f,j,k]
                    dw[f] += segmentA * dout[i,f,j,k]
                    db[f] += dout[i,f,j,k]
    dx = dx_padded[:,:,pad:-pad,pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here. Output size is given by

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    N,C,H,W = x.shape

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    Hprime = int(1 + (H - pool_height) / stride)
    Wprime = int(1 + (W - pool_width) / stride)
    out = np.zeros((N,C,Hprime,Wprime),dtype=float)
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    for i in range(N):
        for c in range(C):
            for h in range(Hprime):
                for w in range(Wprime):
                    xj = h*stride
                    xk = w*stride
                    segmentA = x[i,c,xj:xj+pool_height,xk:xk+pool_width]
                    max_pool = np.max(segmentA)
                    out[i,c,h,w] = max_pool

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    x, pool_param = cache
    dx = None
    N,C,H,W = x.shape

    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    Hprime = int(1 + (H - pool_height) / stride)
    Wprime = int(1 + (W - pool_width) / stride)
    dx = np.zeros((N,C,H,W),dtype=float)
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    for i in range(N):
        for c in range(C):
            for h in range(Hprime):
                for w in range(Wprime):
                    xj = h*stride
                    xk = w*stride
                    segmentA = x[i,c,xj:xj+pool_height,xk:xk+pool_width]
                    max_pool = np.max(segmentA)
                    mask = segmentA>=max_pool
                    dx[i,c,xj:xj+pool_height,xk:xk+pool_width] += dout[i,c,h,w]*mask


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    out=np.zeros_like(x)
    N,C,H,W = x.shape
    for c in range(C):
        x_reshaped = np.reshape(x[:,c,:,:], (N*H*W, 1))
        x_reshaped_batchnorm, _ = batchnorm_forward(x_reshaped, gamma[c], beta[c], bn_param)
        out[:,c,:,:] = np.reshape(x_reshaped_batchnorm, (N,H,W))
    cache = (x, gamma, beta, bn_param)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return out, cache

def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################

    x, gamma, beta, bn_param = cache
    dx = np.zeros_like(x)
    dgamma = np.zeros_like(gamma)
    dbeta = np.zeros_like(beta)
    N, C, H, W = dout.shape
    for c in range(C):
        x_reshaped = np.reshape(x[:,c,:,:], (N*H*W,1))
        dout_reshaped = np.reshape(dout[:,c,:,:], (N*H*W,1))
        dx_reshaped_,dgamma_,dbeta_ = batchnorm_backward_alt(dout_reshaped, (x_reshaped, gamma[c], beta[c], bn_param))
        dx[:,c,:,:] = np.reshape(dx_reshaped_, (N,H,W))
        dgamma[c] = np.sum(dgamma_)
        dbeta[c] = np.sum(dbeta_)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """
    Computes the forward pass for spatial group normalization.
    In contrast to layer normalization, group normalization splits each entry
    in the data into G contiguous pieces, which it then normalizes independently.
    Per feature shifting and scaling are then applied to the data, in a manner identical to that of batch normalization and layer normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get('eps',1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    N,C,H,W = x.shape
    x_reshaped = np.reshape(x, (N,G,C//G,H,W))
    varf_reshaped = np.zeros_like(x_reshaped)
    xc_reshaped = np.zeros_like(x_reshaped)
    for g in range(G):
        x_reshaped_g = x_reshaped[:,g,:,:,:]
        sample_mean = np.mean(x_reshaped_g, axis=1,keepdims=True)
        varf_reshaped[:,g,:,:,:] = np.sqrt(np.var(x_reshaped_g, axis=1,keepdims=True) + eps)
        a=(x_reshaped_g-sample_mean)
        xc_reshaped[:,g,:,:,:] = (a)/varf_reshaped[:,g,:,:,:]
    xc = np.reshape(xc_reshaped, (N,C,H,W))
    varf = np.reshape(varf_reshaped, (N,C,H,W))
    out = xc*gamma + beta
    cache = (x,xc,varf,gamma, beta, G, gn_param)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    x,xc,varf_orig,gamma, beta, G, gn_param = cache
    eps = gn_param.get('eps',1e-5)
    N, C, H, W = dout.shape
    q_orig=xc
    dfdq_orig = dout*gamma
    dfdq_reshaped = np.reshape(dfdq_orig,(N,G,C//G,H,W))
    q_reshaped = np.reshape(q_orig,(N,G,C//G,H,W))
    varf_reshaped = np.reshape(varf_orig,(N,G,C//G,H,W))
    dx_reshaped = np.zeros( (N,G,C//G,H,W))
    for g in range(G):
        dfdq=dfdq_reshaped[:,g,:,:,:]
        q=q_reshaped[:,g,:,:,:]
        varf=varf_reshaped[:,g,:,:,:]
        n=dfdq.shape[1]
        dx_reshaped[:,g,:,:,:]=(n*dfdq - np.sum(dfdq,axis=1,keepdims=True)-q*np.sum(dfdq*q,axis=1,keepdims=True))/(n*varf)

    xcdout_tr = np.reshape(np.transpose(xc*dout,(0,2,3,1)),(N*H*W,C))
    dout_tr = np.reshape(np.transpose(dout,(0,2,3,1)),(N*H*W,C))
    dx = np.reshape(dx_reshaped,(N,C,H,W))
    #dbeta = np.sum(dout, axis=(0,2,3),keepdims=True)
    dgamma = np.zeros_like(gamma)
    dbeta = np.zeros_like(beta)
    dout_trs = np.sum(dout_tr, axis=0)
    xcdout_trs = np.sum(xcdout_tr,axis=0)
    for c in range(C):
        dbeta[:,c,:,:] = dout_trs[c]
        dgamma[:,c,:,:] = xcdout_trs[c]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
