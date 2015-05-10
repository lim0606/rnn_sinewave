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
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#plt.ion()

#mode = theano.Mode(linker='cvm') #the runtime algo to execute the code is in c

def recurrent_fn(y_tm1):
    y_t = y_tm1 + 1
    return y_t

# sequences, prior results(s), non-sequencs
if __name__ == "__main__":
    t0 = time.time()

    #data = np.linspace(1, 10, 10)
    #data = np.reshape(data, (1, 10))
    #print data.shape
    
    Y = T.matrix("Y")
       
    results, updates = theano.scan(fn=lambda y_tm1: y_tm1+1, 
                         outputs_info=[dict(initial=Y, taps = [-1])],
                         n_steps=10)
    pred = theano.function(inputs=[Y], outputs=[results])

    y = np.zeros((1,1), dtype=theano.config.floatX)
    y[0] = 0
    print y
    yy = pred(y)
    yy[0] = yy[0].reshape((10,1))
    print yy

    # define tensor variables
    X = T.matrix("X")
    W = T.matrix("W")
    b_sym = T.vector("b_sym")
    U = T.matrix("U")
    V = T.matrix("V")
    n_sym = T.iscalar("n_sym")

    results, updates = theano.scan(lambda x_tm2, x_tm1: T.dot(x_tm2, U) + T.dot(x_tm1, V) + T.tanh(T.dot(x_tm1, W) + b_sym),
                                   n_steps=n_sym, 
                                   outputs_info=[dict(initial=X, taps=[-2, -1])])
    compute_seq2 = theano.function(inputs=[X, U, V, W, b_sym, n_sym], outputs=[results])

    # test values
    x = np.zeros((2, 1), dtype=theano.config.floatX) # the initial value must be able to return x[-2]
    x[1] = 1
    print x 
    w = 0.5 * np.ones((1, 1), dtype=theano.config.floatX)
    u = 0.5 * (np.ones((1, 1), dtype=theano.config.floatX) - np.eye(1, dtype=theano.config.floatX))
    v = 0.5 * np.ones((1, 1), dtype=theano.config.floatX)
    n = 10
    b = np.ones((1), dtype=theano.config.floatX)

    print compute_seq2(x, u, v, w, b, n)

    # comparison with numpy
    x_res = np.zeros((10, 1))
    x_res[0] = x[0].dot(u) + x[1].dot(v) + np.tanh(x[1].dot(w) + b)
    x_res[1] = x[1].dot(u) + x_res[0].dot(v) + np.tanh(x_res[0].dot(w) + b)
    x_res[2] = x_res[0].dot(u) + x_res[1].dot(v) + np.tanh(x_res[1].dot(w) + b)
    for i in range(2, 10):
        x_res[i] = (x_res[i - 2].dot(u) + x_res[i - 1].dot(v) + np.tanh(x_res[i - 1].dot(w) + b))
    print x_res
   
    print "Elapsed time: %f" % (time.time() - t0)

