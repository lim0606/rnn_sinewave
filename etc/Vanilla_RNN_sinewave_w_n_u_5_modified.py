"""

Vanilla Recurrent Neural Network
Code provided by Mohammad Pezeshki - Nov. 2014 - Universite de Montreal
This code is distributed without any warranty, express or implied. 
Thanks to Razvan Pascanu and Graham Taylor for their codes available at:
https://github.com/pascanur/trainingRNNs
https://github.com/gwtaylor/theano-rnn

"""

import numpy as np
import theano
import theano.tensor as T
import time
import os
import datetime
import matplotlib

import collections
import math

# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

mode = theano.Mode(linker='cvm') #the runtime algo to execute the code is in c


"""
What we have in this class:

    Model structure parameters:
        n_u : length of input layer vector in each time-step
        n_h : length of hidden layer vector in each time-step
        n_y : length of output layer vector in each time-step
        activation : type of activation function used for hidden layer
        output_type : type of output which could be `real`, `binary`, or `softmax`

    Parameters to be learned:
        W_uh : weight matrix from input to hidden layer
        W_hh : recurrent weight matrix from hidden to hidden layer
        W_hy : weight matrix from hidden to output layer
        b_h : biases vector of hidden layer
        b_y : biases vector of output layer
        h0 : initial values for the hidden layer

    Learning hyper-parameters:
        learning_rate : learning rate which is not constant
        learning_rate_decay : learning rate decay :)
        L1_reg : L1 regularization term coefficient
        L2_reg : L2 regularization term coefficient
        initial_momentum : momentum value which we start with
        final_momentum : final value of momentum
        momentum_switchover : on which `epoch` should we switch from
                              initial value to final value of momentum
        n_epochs : number of iterations

    Inner class variables:
        self.x : symbolic input vector
        self.y : target output
        self.y_pred : raw output of the model
        self.p_y_given_x : output after applying sigmoid (binary output case)
        self.y_out : round (0,1) for binary and argmax (0,1,...,k) for softmax
        self.loss : loss function (MSE or CrossEntropy)
        self.predict : a function returns predictions which is type is related to output type
        self.predict_proba : a function returns predictions probabilities (binary and softmax)
    
    build_train function:
        train_set_x : input of network
        train_set_y : target of network
        index : index over each ...............................
        lr : learning rate
        mom : momentum
        cost : cost function value
        compute_train_error : a function compute error on training
        gparams : Gradients of model parameters
        updates : updates which should be applied to parameters
        train_model : a function that returns the cost, but 
                      in the same time updates the parameter
                      of the model based on the rules defined
                      in `updates`.
        
"""
class RNN(object):
    def __init__(self, n_u, n_h, n_y, activation, output_type,
                 learning_rate, learning_rate_decay, L1_reg, L2_reg,
                 initial_momentum, final_momentum, momentum_switchover,
                 n_epochs):

        self.n_u = int(n_u)
        self.n_h = int(n_h)
        self.n_y = int(n_y)

        if activation == 'tanh':
            self.activation = T.tanh
        elif activation == 'sigmoid':
            self.activation = T.nnet.sigmoid
        elif activation == 'relu':
            self.activation = lambda x: x * (x > 0) # T.maximum(x, 0)
        else:
            raise NotImplementedError   

        self.output_type = output_type
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        self.n_epochs = int(n_epochs)

        # input which is `x`
        self.x = T.matrix()

        # weights are initialized from an uniform distribution
        self.W_uh = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_u, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_uh')

        self.W_hh = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_h),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hh')

        self.W_hy = theano.shared(value = np.asarray(
                                              np.random.uniform(
                                                  size = (n_h, n_y),
                                                  low = -.01, high = .01),
                                              dtype = theano.config.floatX),
                                  name = 'W_hy')

        # initial value of hidden layer units are set to zero
        self.h0 = theano.shared(value = np.zeros(
                                            (n_h, ),
                                            dtype = theano.config.floatX),
                                name = 'h0')

        # biases are initialized to zeros
        self.b_h = theano.shared(value = np.zeros(
                                             (n_h, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_h')

        self.b_y = theano.shared(value = np.zeros(
                                             (n_y, ),
                                             dtype = theano.config.floatX),
                                 name = 'b_y')

        self.params = [self.W_uh, self.W_hh, self.W_hy, self.h0,
                       self.b_h, self.b_y]

        # Initial value for updates is zero matrix.
        #self.updates = {}
        self.updates = collections.OrderedDict()
        for param in self.params:
            self.updates[param] = theano.shared(
                                      value = np.zeros(
                                                  param.get_value(
                                                      borrow = True).shape,
                                                      dtype = theano.config.floatX),
                                      name = 'updates')

        # h_t = g(W_uh * u_t + W_hh * h_tm1 + b_h)
        # y_t = W_yh * h_t + b_y
        def recurrent_fn(u_t, h_tm1):
            h_t = self.activation(T.dot(u_t, self.W_uh) + \
                                  T.dot(h_tm1, self.W_hh) + \
                                  self.b_h)
            y_t = T.dot(h_t, self.W_hy) + self.b_y
            return h_t, y_t

        # Iteration over the first dimension of a tensor which is TIME in our case
        # recurrent_fn doesn't use y in the computations, so we do not need y0 (None)
        # scan returns updates too which we do not need. (_)
        [self.h, self.y_pred], _ = theano.scan(recurrent_fn,
                                               sequences = self.x,
                                               outputs_info = [self.h0, None])

        # L1 norm
        self.L1 = abs(self.W_uh.sum()) + \
                  abs(self.W_hh.sum()) + \
                  abs(self.W_hy.sum())

        # square of L2 norm
        self.L2_sqr = (self.W_uh ** 2).sum() + \
                      (self.W_hh ** 2).sum() + \
                      (self.W_hy ** 2).sum()

        # Loss function is different for different output types
        # defining function in place is so easy! : lambda input: expresion
        if self.output_type == 'real':
            self.y = T.matrix(name = 'y', dtype = theano.config.floatX)
            self.loss = lambda y: self.mse(y) # y is input and self.mse(y) is output
            self.predict = theano.function(inputs = [self.x, ],
                                           outputs = self.y_pred,
                                           mode = mode,
                                           allow_input_downcast=True)

        elif self.output_type == 'binary':
            self.y = T.matrix(name = 'y', dtype = 'int32')
            self.p_y_given_x = T.nnet.sigmoid(self.y_pred)
            self.y_out = T.round(self.p_y_given_x)  # round to {0,1}
            self.loss = lambda y: self.nll_binary(y)
            self.predict_proba = theano.function(inputs = [self.x, ],
                                                 outputs = self.p_y_given_x,
                                                 mode = mode)
            self.predict = theano.function(inputs = [self.x, ],
                                           outputs = T.round(self.p_y_given_x),
                                           mode = mode,
                                           allow_input_downcast=True)
        
        elif self.output_type == 'softmax':
            self.y = T.vector(name = 'y', dtype = 'int32')
            self.p_y_given_x = T.nnet.softmax(self.y_pred)
            self.y_out = T.argmax(self.p_y_given_x, axis = -1)
            self.loss = lambda y: self.nll_multiclass(y)
            self.predict_proba = theano.function(inputs = [self.x, ],
                                                 outputs = self.p_y_given_x,
                                                 mode = mode)
            self.predict = theano.function(inputs = [self.x, ],
                                           outputs = self.y_out, # y-out is calculated by applying argmax
                                           mode = mode,
                                           allow_input_downcast=True)
        else:
            raise NotImplementedError

        # Just for tracking training error for Graph 3
        self.errors = []

    def mse(self, y):
        # mean is because of minibatch
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood here is cross entropy
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def nll_multiclass(self, y):
        # notice to [  T.arange(y.shape[0])  ,  y  ]
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    # X_train, Y_train, X_test, and Y_test are numpy arrays
    def build_train(self, X_train, Y_train, X_test = None, Y_test = None):
        train_set_x = theano.shared(np.asarray(X_train, dtype=theano.config.floatX))
        train_set_y = theano.shared(np.asarray(Y_train, dtype=theano.config.floatX))
        if self.output_type in ('binary', 'softmax'):
            train_set_y = T.cast(train_set_y, 'int32')

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        print 'Buiding model ...'

        index = T.lscalar('index')    # index to a case
        # learning rate (may change)
        lr = T.scalar('lr', dtype = theano.config.floatX)
        mom = T.scalar('mom', dtype = theano.config.floatX)  # momentum


        # Note that we use cost for training
        # But, compute_train_error for just watching
        cost = self.loss(self.y) \
            + self.L1_reg * self.L1 \
            + self.L2_reg * self.L2_sqr

        # We don't want to pass whole dataset every time we use this function.
        # So, the solution is to put the dataset in the GPU as `givens`.
        # And just pass index to the function each time as input.
        compute_train_error = theano.function(inputs = [index, ],
                                              outputs = self.loss(self.y),
                                              givens = {
                                                  self.x: train_set_x[index],
                                                  self.y: train_set_y[index]},
                                              mode = mode,
                                              allow_input_downcast=True)

        # Gradients of cost wrt. [self.W, self.W_in, self.W_out,
        # self.h0, self.b_h, self.b_y] using BPTT.
        gparams = []
        for param in self.params:
            gparams.append(T.grad(cost, param))

        # zip just concatenate two lists
        #updates = {}
        updates = collections.OrderedDict()
        for param, gparam in zip(self.params, gparams):
            weight_update = self.updates[param]
            upd = mom * weight_update - lr * gparam
            updates[weight_update] = upd
            updates[param] = param + upd

        # compiling a Theano function `train_model` that returns the
        # cost, but in the same time updates the parameter of the
        # model based on the rules defined in `updates`
        train_model = theano.function(inputs = [index, lr, mom],
                                      outputs = cost,
                                      updates = updates,
                                      givens = {
                                          self.x: train_set_x[index], # [:, batch_start:batch_stop]
                                          self.y: train_set_y[index]},
                                      mode = mode,
                                      allow_input_downcast=True)

        ###############
        # TRAIN MODEL #
        ###############
        print 'Training model ...'
        epoch = 0
        n_train = train_set_x.get_value(borrow = True).shape[0]

        while (epoch < self.n_epochs):
            epoch = epoch + 1
            indices = np.random.randint(0, n_train, size=16) 
            #for idx in xrange(n_train):
            for idx in indices:
                effective_momentum = self.final_momentum \
                                     if epoch > self.momentum_switchover \
                                     else self.initial_momentum
                example_cost = train_model(idx,
                                           self.learning_rate,
                                           effective_momentum)

            # compute loss on training set
            train_losses = [compute_train_error(i)
                            for i in xrange(n_train)]
            this_train_loss = np.mean(train_losses)
            self.errors.append(this_train_loss)

            print('epoch %i, train loss %f ''lr: %f' % \
                  (epoch, this_train_loss, self.learning_rate))

            self.learning_rate *= self.learning_rate_decay
"""
Here we define some testing functions.
For more details see Graham Taylor model:
https://github.com/gwtaylor/theano-rnn
"""
"""
Here we test the RNN with real output.
We randomly generate `n_seq` sequences of length `time_steps`.
Then we make a delay to get the targets. (+ adding some noise)
Resulting graphs are saved under the name of `real.png`.
"""
def test_real():
    print 'Testing model with real outputs'
    n_u = 3 # input vector size (not time at this point)
    n_h = 10 # hidden vector size
    n_y = 3 # output vector size
    time_steps = 15 # number of time-steps in time
    n_seq = 100 # number of sequences for training

    np.random.seed(0)
    
    # generating random sequences
    seq = np.random.randn(n_seq, time_steps, n_u)
    targets = np.zeros((n_seq, time_steps, n_y))

    targets[:, 1:, 0] = seq[:, :-1, 0] # 1 time-step delay between input and output
    targets[:, 2:, 1] = seq[:, :-2, 1] # 2 time-step delay
    targets[:, 3:, 2] = seq[:, :-3, 2] # 3 time-step delay

    targets += 0.01 * np.random.standard_normal(targets.shape)

    model = RNN(n_u = n_u, n_h = n_h, n_y = n_y,
                activation = 'tanh', output_type = 'real',
                learning_rate = 0.001, learning_rate_decay = 0.999,
                L1_reg = 0, L2_reg = 0, 
                initial_momentum = 0.5, final_momentum = 0.9,
                momentum_switchover = 5,
                n_epochs = 400)

    model.build_train(seq, targets)


    # We just plot one of the sequences
    plt.close('all')
    fig = plt.figure()

    # Graph 1
    ax1 = plt.subplot(311) # numrows, numcols, fignum
    plt.plot(seq[0])
    plt.grid()
    ax1.set_title('Input sequence')

    # Graph 2
    ax2 = plt.subplot(312)
    true_targets = plt.plot(targets[0])

    guess = model.predict(seq[0])
    guessed_targets = plt.plot(guess, linestyle='--')
    plt.grid()
    for i, x in enumerate(guessed_targets):
        x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')

    # Graph 3
    ax3 = plt.subplot(313)
    plt.plot(model.errors)
    plt.grid()
    ax1.set_title('Training error')

    # Save as a file
    plt.savefig('real_5.png')

"""
Here we test the RNN with binary output.
We randomly generate `n_seq` sequences of length `time_steps`.
Then we make a delay and make binary number which are obtained 
using comparison to get the targets. (+ adding some noise)
Resulting graphs are saved under the name of `binary.png`.
"""
def test_binary():
    print 'Testing model with binary outputs'
    n_u = 2
    n_h = 5
    n_y = 1
    time_steps = 20
    n_seq = 100

    np.random.seed(0)

    seq = np.random.randn(n_seq, time_steps, n_u)
    targets = np.zeros((n_seq, time_steps, n_y))

    # whether `dim 3` is greater than `dim 0`
    targets[:, 2:, 0] = np.cast[np.int](seq[:, 1:-1, 1] > seq[:, :-2, 0])

    model = RNN(n_u = n_u, n_h = n_h, n_y = n_y,
                activation = 'tanh', output_type = 'binary',
                learning_rate = 0.001, learning_rate_decay = 0.999,
                L1_reg = 0, L2_reg = 0, 
                initial_momentum = 0.5, final_momentum = 0.9,
                momentum_switchover = 5,
                n_epochs = 700)

    model.build_train(seq, targets)

    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(seq[1])
    plt.grid()
    ax1.set_title('input')
    ax2 = plt.subplot(312)
    guess = model.predict_proba(seq[1])
    # put target and model output beside each other
    plt.imshow(np.hstack((targets[1], guess)).T, interpolation = 'nearest', cmap = 'gray')

    plt.grid()
    ax2.set_title('first row: true output, second row: model output')

    ax3 = plt.subplot(313)
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')

    plt.savefig('binary_5.png')

"""
Here we test the RNN with softmax output.
We randomly generate `n_seq` sequences of length `time_steps`.
Then we make a delay and make classed which are obtained 
using comparison to get the targets.
Resulting graphs are saved under the name of `softmax.png`.
"""
def test_softmax():
    print 'Testing model with softmax outputs'
    n_u = 2
    n_h = 6
    n_y = 3 # equal to the number of calsses
    time_steps = 10
    n_seq = 100

    np.random.seed(0)

    seq = np.random.randn(n_seq, time_steps, n_u)
    # Note that is this case `targets` is a 2d array
    targets = np.zeros((n_seq, time_steps), dtype=np.int)

    thresh = 0.5
    # Comparisons to assing a class label in output
    targets[:, 2:][seq[:, 1:-1, 1] > seq[:, :-2, 0] + thresh] = 1
    targets[:, 2:][seq[:, 1:-1, 1] < seq[:, :-2, 0] - thresh] = 2
    # otherwise class is 0

    model = RNN(n_u = n_u, n_h = n_h, n_y = n_y,
                activation = 'tanh', output_type = 'softmax',
                learning_rate = 0.001, learning_rate_decay = 0.999,
                L1_reg = 0, L2_reg = 0, 
                initial_momentum = 0.5, final_momentum = 0.9,
                momentum_switchover = 5,
                n_epochs = 500)

    model.build_train(seq, targets)

    plt.close('all')
    fig = plt.figure()
    ax1 = plt.subplot(311)
    plt.plot(seq[1])
    plt.grid()
    ax1.set_title('input')
    ax2 = plt.subplot(312)

    plt.scatter(xrange(time_steps), targets[1], marker = 'o', c = 'b')
    plt.grid()

    guess = model.predict_proba(seq[1])
    guessed_probs = plt.imshow(guess.T, interpolation = 'nearest', cmap = 'gray')
    ax2.set_title('blue points: true class, grayscale: model output (white mean class)')

    ax3 = plt.subplot(313)
    plt.plot(model.errors)
    plt.grid()
    ax3.set_title('Training error')
    plt.savefig('softmax_5.png')


if __name__ == "__main__":
    t0 = time.time()
    #test_real()
    #test_binary()
    #test_softmax()

    print 'Generate sine wave dataset'
    n_sinewave = 1000 #20000
    #n_train_sinewave = n_sinewave # number of sine wave sequences (train)
    #n_test_sinewave = n_sinewave # number of sine wave sequences (test)
    length_sinewave = 160 # length of each sine wave sequence
   
    seq_sine_alpha_train = np.random.uniform(1, 5, (n_sinewave, 1))
    seq_sine_alpha_test = np.random.uniform(1, 5, (n_sinewave, 1))
    seq_sine_beta_train = np.random.uniform(-1, 1, (n_sinewave, 1))
    seq_sine_beta_test = np.random.uniform(-1, 1, (n_sinewave, 1))
    
    print seq_sine_alpha_train.shape
    print seq_sine_beta_train.shape

    sinewave = np.linspace(0, 2 * math.pi, length_sinewave)
    sinewave_train = np.zeros((n_sinewave, length_sinewave))
    sinewave_test = np.zeros((n_sinewave, length_sinewave))
    
    print sinewave.shape

    for i in range(n_sinewave):
        sinewave_train[i,:] = np.sin(seq_sine_alpha_train[i] * sinewave + seq_sine_beta_train[i] * 2 * math.pi)
    for i in range(n_sinewave):
        sinewave_test[i,:] = np.sin(seq_sine_alpha_test[i] * sinewave + seq_sine_beta_test[i] * 2 * math.pi)

  
    plt.close('all')
    fig_sinewave = plt.figure()
    input_seq1 = plt.plot(sinewave, sinewave_train[0,:], label='input seq1')
    input_seq2 = plt.plot(sinewave, sinewave_test[0,:], label='input seq2')
    plt.title('sample input sequences')
    plt.legend()  
    plt.savefig('sinewave_5.png')

    
    print 'Testing model with real outputs'
    n_u = 5 # input vector size (not time at this point)
    n_h = 10 # hidden vector size
    n_y = 1 # output vector size
    time_steps = length_sinewave - n_u # number of time-steps in time
    n_seq = n_sinewave # number of sequences for training

    #np.random.seed(0)
    
    # generating random sequences
    seq = np.zeros((n_seq, time_steps, n_u))
    targets = np.zeros((n_seq, time_steps, n_y))
    seq_test = np.zeros((n_seq, time_steps, n_u))
    targets_test = np.zeros((n_seq, time_steps, n_y))

    """
    targets[:, 1:, 0] = seq[:, :-1, 0] # 1 time-step delay between input and output
    targets[:, 2:, 1] = seq[:, :-2, 1] # 2 time-step delay
    targets[:, 3:, 2] = seq[:, :-3, 2] # 3 time-step delay
    """
    ## overlapping case
    # train
    for i in range(n_seq):
        for j in range(time_steps):
            seq[i, j, :] = sinewave_train[i,j:j+n_u]
            targets[i, j, :] = sinewave_train[i,j+n_u]
    # test
    for i in range(n_seq):
        for j in range(time_steps):
            seq_test[i, j, :] = sinewave_test[i,j:j+n_u]
            targets_test[i, j, :] = sinewave_test[i,j+n_u]

    plt.close('all')
    fig_data = plt.figure()
    input_seq = plt.plot(np.linspace(1, seq.shape[1], seq.shape[1]), seq[0,:,:], label='u_t, input seq')
    output = plt.plot(np.linspace(1, seq.shape[1], seq.shape[1]), targets[0,:,:], linestyle='--', label='y_t, output')
    plt.title('sample input seq(u_t) and its output(y_t)')
    plt.legend()
    plt.savefig('data_5.png')
 
    #targets += 0.01 * np.random.standard_normal(targets.shape)

    model = RNN(n_u = n_u, n_h = n_h, n_y = n_y,
                activation = 'tanh', output_type = 'real',
                learning_rate = 0.001, learning_rate_decay = 0.999,
                L1_reg = 0, L2_reg = 0, 
                initial_momentum = 0.5, final_momentum = 0.9,
                momentum_switchover = 5,
                n_epochs = 400)

    model.build_train(seq, targets)

    #######
    seq_idx = 10

    # We just plot one of the sequences
    plt.close('all')
    fig = plt.figure()
    '''
    # Graph 1
    ax1 = plt.subplot(311) # numrows, numcols, fignum
    plt.plot(seq_test[6,:,:])
    plt.grid()
    ax1.set_title('Input sequence')

    # Graph 2
    ax2 = plt.subplot(312)
    plt.plot(seq_test[6,:,:])
    true_targets = plt.plot(targets_test[6,:,:],':')

    guess = model.predict(seq_test[6,:,:])
    guessed_targets = plt.plot(guess, linestyle='--')
    plt.grid()
    #for i, x in enumerate(guessed_targets):
    #    x.set_color(true_targets[i].get_color())
    ax2.set_title('solid: true output, dashed: model output')

    # Graph 3
    ax3 = plt.subplot(313)
    plt.plot(model.errors)
    plt.grid()
    ax1.set_title('Training error')
    '''
    plt.plot(model.errors)
    plt.grid()
    plt.title('Training error')

    # Save as a file
    plt.savefig('real_5.png')


    t1 = time.time()
    print "Elapsed time: %f" % (t1 - t0)

    #########################################################################################################
    # predict!! from 50 frames! predict 110 frames :)
    #print seq_test[seq_idx,0:50,:]
    #print targets[seq_idx,:,:]
    W_uh = theano.shared(value = model.W_uh.get_value(), name = 'W_uh') #matrix
    W_hh = theano.shared(value = model.W_hh.get_value(), name = 'W_hh') #matrix
    W_hy = theano.shared(value = model.W_hy.get_value(), name = 'W_hy') #matrix
    h0 = theano.shared(value = np.zeros((n_h, ), dtype = theano.config.floatX), name = 'h0') #vector
    b_h = theano.shared(value = model.b_h.get_value(), name = 'b_h') #vector
    b_y = theano.shared(value = model.b_y.get_value(), name = 'b_y') #vector

    def recurrent_fn1(u_t, h_tm1):
        h_t = T.tanh(T.dot(u_t, W_uh) + \
                              T.dot(h_tm1, W_hh) + \
                              b_h)
        y_t = T.dot(h_t, W_hy) + b_y
        return h_t, y_t
 
    u_t = T.matrix('u_t')

    [h_t, y_t],_ = theano.scan(fn=recurrent_fn1, 
                sequences=u_t,
                outputs_info=[h0, None])
    predict1 = theano.function(inputs=[u_t,],
                               outputs=[h_t, y_t],
                               allow_input_downcast=True)

    [h_t_val, y_t_val] = predict1(seq_test[seq_idx,0:50,:])

    '''def recurrent_fn2(h_tm1, y_tm5, y_tm4, y_tm3, y_tm2, y_tm1):
        h_t = T.tanh(T.dot(T.concatenate([y_tm5, y_tm4, y_tm3, y_tm2, y_tm1], axis=1), W_uh) + \
                              T.dot(h_tm1, W_hh) + \
                              b_h)
        y_t = T.dot(h_t, W_hy) + b_y
        return h_t, y_t'''

    previous_time = 5

    def recurrent_fn2(h_tm1, *y_tms):
        y_tms_concat = y_tms[0]
        for y_tm in y_tms[1:]:
            y_tms_concat = T.concatenate([y_tms_concat, y_tm], axis=1)

        h_t = T.tanh(T.dot(y_tms_concat, W_uh) + \
                          T.dot(h_tm1, W_hh) + \
                          b_h)
        y_t = T.dot(h_t, W_hy) + b_y
        return h_t, y_t

    H_t = T.matrix("H_t")
    Y_t = T.tensor3("Y_t")

    y_t_pred_taps = []
    for t in xrange(-previous_time, 0, 1):
        y_t_pred_taps.append(t)

    #[h_t_pred, y_t_pred],_ = theano.scan(fn=recurrent_fn2,
    #                                     outputs_info=[dict(initial=H_t, taps=[-1]), dict(initial=Y_t, taps=[-5, -4, -3, -2, -1])],
    #  
    [h_t_pred, y_t_pred],_ = theano.scan(fn=recurrent_fn2,
                                         outputs_info=[dict(initial=H_t, taps=[-1]), dict(initial=Y_t, taps=y_t_pred_taps)],
                                        n_steps=110)
    predict2 = theano.function(inputs=[H_t, Y_t],
                               outputs=[h_t_pred, y_t_pred], 
                               allow_input_downcast=True)
    y_t_pred0 = np.reshape(np.concatenate((seq_test[seq_idx,50-1,1:n_u], y_t_val[-1])), (n_u,1,1))
    h_t_pred0 = np.reshape(h_t_val[-1], (1,n_h))
    
    [h_t_pred_val, y_t_pred_val] = predict2(h_t_pred0, y_t_pred0)
    y_t_pred_val = np.reshape(y_t_pred_val, (110, 1))

    y_t_pred_total = np.concatenate((y_t_val, y_t_pred_val), axis=0)

    # We just plot one of the sequences
    plt.close('all')
    fig_pred = plt.figure()

    # Graph 1
    #plt.plot(seq_test[seq_idx,:,:])
    true_targets, = plt.plot(targets_test[seq_idx,:,:], label='y_t, true ouput')
    guessed_targets, = plt.plot(y_t_pred_total, linestyle='--', label='y_t_pred, model output')
    verticalline = plt.axvline(x=49, color='r', linestyle=':')
    plt.grid()
    plt.legend()
    plt.ylim((-3, 3))
    plt.title('true output vs. model output')

    # Save as a file
    plt.savefig('pred_5.png')

    t2 = time.time()
    print "Elapsed time: %f" % (t2 - t1)

    print "Total Elapsed time: %f" % (t2 - t0)

 
