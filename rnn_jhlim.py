'''
This code is cutomized to use sinewave generation from lstm.py provided in http://deeplearning.net/tutorial/lstm.html
'''
from collections import OrderedDict
import cPickle as pkl
import random
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

#import imdb
import sinewave 

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

#datasets = {'imdb': (imdb.load_data, imdb.prepare_data)}
datasets = {'sinewave': (sinewave.load_data, sinewave.prepare_data)}


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
                                    minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def get_dataset(name):
    return datasets[name][0], datasets[name][1]


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=0.5, n=1,
                                        dtype=state_before.dtype)),
                         state_before * 0.5)
    return proj


def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    randn = numpy.random.rand(options['input_dim'],  # 10000
                              options['dim_proj']) # 128
    params['Wemb'] = (0.01 * randn).astype(config.floatX) # 10000 x 128 matrix
    params = get_layer(options['encoder'])[0](options,    #param_init_lstm
                                              params,
                                              prefix=options['encoder'])
    # params now has following objects
    #        params[Wemb], params[lstm_W], params[lstm_U], params[lstm_b] 

    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['dim_proj'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)
    # params now has following objects                                            #        params[Wemb], params[lstm_W], params[lstm_U], params[lstm_b],
    #                                      params[U], params[b]
    # params with prefix lstm_ have initialzed with ortho_weight function
    # why does this code initialize twice? with different method? 

    return params


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            raise Warning('%s is not in the archive' % kk)
        params[kk] = pp[kk]

    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def get_layer(name):
    fns = layers[name]
    return fns


def ortho_weight(ndim):
    W = 0.01 * numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(config.floatX)


def uniform_weight(ndim):
    W = numpy.random.uniform(size = (ndim, ndim),
                             low = -.01, high = .01)
    return W.astype(config.floatX)


def param_init_lstm(options, params, prefix='lstm'):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    '''
    W = numpy.concatenate([ortho_weight(options['dim_proj']),  # dim_proj = 128
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)'''
    W = numpy.concatenate([options['initializer'](options['dim_proj']),  # dim_proj = 128
                           options['initializer'](options['dim_proj']),
                           options['initializer'](options['dim_proj']),
                           options['initializer'](options['dim_proj'])], axis=1)
    params[_p(prefix, 'W')] = W
    '''U = numpy.concatenate([ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj']),
                           ortho_weight(options['dim_proj'])], axis=1)'''
    U = numpy.concatenate([options['initializer'](options['dim_proj']), 
                           options['initializer'](options['dim_proj']),
                           options['initializer'](options['dim_proj']),
                           options['initializer'](options['dim_proj'])], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * options['dim_proj'],))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def param_init_rnn(options, params, prefix='rnn'):
    """
    Init the RNN parameter:

    :see: init_params
    """
    U = options['initializer'](options['dim_proj']) #uniform_weight(options['dim_proj']) #ortho_weight(options['dim_proj']) # dim_proj = 128
    params[_p(prefix, 'U')] = U
    b = numpy.zeros(options['dim_proj'])
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params

def lstm_layer(tparams, state_below, options, prefix='lstm', mask=None):
    # state_below = input = emb = n_timesteps x n_samples x 128
    nsteps = state_below.shape[0] # state_below = emb, state_below.shape[0] = x.shape[0] = n_timesteps
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_): # m_: mask, x_: input, h_: previous hidden, c_: previous hidden
        #print "x_.shape : ", x_.shape
        #print "x_.ndim : ", x_.ndim 
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        preact += tparams[_p(prefix, 'b')]

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0], [rval[1]] # rval[0] = h, rval[1] = c, this task requires h of lstm layer. 


def rnn_layer(tparams, state_below, options, prefix='rnn', mask=None):
    # state_below = input = emb = n_timesteps x n_samples x 128
    nsteps = state_below.shape[0] # state_below = emb, state_below.shape[0] = x.shape[0] = n_timesteps
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_): # m_: mask, x_: input, h_: previous hidden
        #print "x_.shape : ", x_.shape
        #print "x_.ndim : ", x_.ndim 
        h = tensor.dot(h_, tparams[_p(prefix, 'U')])
        h += x_
        h += tparams[_p(prefix, 'b')]
        h = tensor.tanh(h)
        return h

    dim_proj = options['dim_proj']
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval, None # rval[0] = h 


# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.
layers = {'lstm': (param_init_lstm, lstm_layer), 
          'rnn': (param_init_rnn, rnn_layer)}


def sgd(lr, tparams, grads, x, mask, y, cost):
    """ Stochastic Gradient Descent

    :note: A more complicated version of sgd then needed.  This is
        done like that for adadelta and rmsprop.

    """
    # New set of shared variable that will contain the gradient
    # for a mini-batch.
    gshared = [theano.shared(p.get_value() * 0.5, name='%s_grad' % k)
               for k, p in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    # Function that computes gradients for a mini-batch, but do not
    # updates the weights.
    f_grad_shared = theano.function([x, mask, y], cost, updates=gsup,
                                    name='sgd_f_grad_shared')

    pup = [(p, p - lr * g) for p, g in zip(tparams.values(), gshared)]

    # Function that updates the weights from the previously computed
    # gradient.
    f_update = theano.function([lr], [], updates=pup,
                               name='sgd_f_update')

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost, updates=zgup + rg2up,
                                    name='adadelta_f_grad_shared')

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads,
                                     running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(tparams.values(), updir)]

    f_update = theano.function([lr], [], updates=ru2up + param_up,
                               on_unused_input='ignore',
                               name='adadelta_f_update')

    return f_grad_shared, f_update


def rmsprop(lr, tparams, grads, x, mask, y, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                   name='%s_rgrad' % k)
                     for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function([x, mask, y], cost,
                                    updates=zgup + rgup + rg2up,
                                    name='rmsprop_f_grad_shared')

    updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                           name='%s_updir' % k)
             for k, p in tparams.iteritems()]
    updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                            running_grads2)]
    param_up = [(p, p + udn[1])
                for p, udn in zip(tparams.values(), updir_new)]
    f_update = theano.function([lr], [], updates=updir_new + param_up,
                               on_unused_input='ignore',
                               name='rmsprop_f_update')

    return f_grad_shared, f_update


def build_model(tparams, options):
    trng = RandomStreams(1234)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.tensor3('x', dtype=config.floatX)
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.tensor3('y', dtype=config.floatX)

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]
    input_dim = x.shape[2]

    emb = (tensor.dot(x.reshape([n_timesteps * n_samples, input_dim]), tparams['Wemb'])).reshape([n_timesteps, n_samples, options['dim_proj']])
    '''emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,          # x.shape[0]
                                                n_samples,            # x.shape[1]
                                                options['dim_proj']]) # 128'''
    f_emb = theano.function(inputs=[x], outputs=emb)
    # emb is word embedding... Thus, this becomes the input to the lstm
    # n_timesteps x n_samples x 128
    # x.flatten() is sequence of indices of words with length n_timesteps x n_samples
    # tparams['Wemb'][x.flatten()] is (n_timesteps x n_samples) x 128

    prefix=options['encoder']
    #print 'tparams[_p(prefix, 'U')].ndim: ', tparams[_p(prefix, 'U')].ndim

    proj, hiddens = get_layer(options['encoder'])[1](tparams, emb, options,    # lstm_layer
                                            prefix=options['encoder'],
                                            mask=mask)
    f_hiddens = []
    f_hiddens.append(theano.function(inputs=[x, mask], outputs=proj))
    if hiddens is not None:
        for f_hidden in hiddens:
            f_hiddens.append(theano.function(inputs=[x, mask], outputs=f_hidden))
    #print "len(f_hiddens): %d" % len(f_hiddens)

    '''if options['encoder'] == 'lstm':
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]'''
    # what is mask??????
    # input x and mask has size of maxlen x n_samples. 
    # lstm_layer assumes that emb has n_timesteps = maxlen. 
    # irrelavant activities of proj should be ignored. so, mask is used for that. 
    # Then, why proj has summed up?
    #       => since the model we assumed multiply 'U' with mean pulled hidden activities.   
    
    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng)
    f_proj = theano.function(inputs=[x, mask], outputs=proj)

    #pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])
    pred = tensor.dot(proj, tparams['U']) + tparams['b'] 
    #print tparams['U'].get_value().shape
    #f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    #f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')
    f_pred = theano.function([x, mask], pred, name='f_pred')

    #cost = -tensor.log(pred[tensor.arange(n_samples), y] + 1e-8).mean()
           # pred[tensor.arange(n_samples), y] ???
           # == pred[tensor.arange(n_samples)][y]
           # multiclass log loss 
    cost = (pred - y) ** 2 # mean squared error
    cost = (cost * mask[:, :, None]).sum(axis=0)
    cost = cost / mask.sum(axis=0)[:, None]
    cost = theano.tensor.mean(cost)

    #return use_noise, x, mask, y, f_pred_prob, f_pred, cost
    #return use_noise, x, mask, y, f_pred, cost, f_pred_h_and_c, f_emb, f_proj
    return use_noise, x, mask, y, f_pred, cost, f_hiddens

def pred_error(f_pred, prepare_data, input_dim, output_dim, data, iterator, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  input_dim,
                                  output_dim,  
                                  maxlen=None)
        preds = f_pred(x, mask)
        #targets = numpy.array(data[1])[valid_index]
        #valid_err += (preds == targets).sum()
        cost = (preds - y) ** 2 # mean squared error
        cost = (cost * mask[:, :, None]).sum(axis=0)
        cost = cost / mask.sum(axis=0)[:, None]
        cost = cost.mean()
        valid_err += cost 
    #valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    valid_err = valid_err / len(iterator)
    return valid_err

def build_lstm_pred_model(tparams, options, n_timesteps=1, input_dim=1):
    trng = RandomStreams(1234)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    use_noise.set_value(0.)

    prefix=options['encoder']

    #print "part1"

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    #def _step(m_, x_, h_, c_): # m_: mask, x_: input, h_: previous hidden, c_: previous hidden
    def _step(h_, c_, *y_): # m_: mask, h_: previous hidden, c_: previous hidden, *y_: raw inputs
        #print "h_.ndim: ", h_.ndim
        #print "c_.ndim: ", c_.ndim
        
        #print "len(y_): ", len(y_)
        #print "y_[0].shape: ", y_[0].shape # y_[0] has size of n_samples x output_dim (output_dim = 1 for sinewave example)
        #print "y_[0].ndim: ", y_[0].ndim

        # build x_ from y_
        x_ = y_[0]
        for y_tmp in y_[1:]:
            x_ = tensor.concatenate([x_, y_tmp], axis=1)

        #print theano.pp(x_)
        #h_printed_ = theano.printing.Print('h_ in step')(h_)
        #c_printed_ = theano.printing.Print('c_ in step')(c_) 
        #x_printed_ = theano.printing.Print('x_ in step')(x_)

        #print "x_.ndim: ", x_.ndim
        # embedding and activation
        emb_ = tensor.dot(x_, tparams['Wemb'])
        #emb_ = tensor.dot(x_printed_, tparams['Wemb']) 
        #print "emb_.ndim: ", emb_.ndim

        #emb_printed_ = theano.printing.Print('emb_ in step')(emb_)

        state_below_ = (tensor.dot(emb_, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])
        #state_below_ = (tensor.dot(emb_printed_, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')])
        #print "state_below_.ndim: ", state_below_.ndim
        #print "tparams[lstm_W].ndim: ", tparams['lstm_W'].ndim
        #state_below_printed_ = theano.printing.Print('state_below_ in step')(state_below_)

         
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        #preact = tensor.dot(h_printed_, tparams[_p(prefix, 'U')])
        preact += state_below_ # x_
        #preact += state_below_printed_ # x_
        preact += tparams[_p(prefix, 'b')]
        #print "preact.ndim: ", preact.ndim

        i = tensor.nnet.sigmoid(_slice(preact, 0, options['dim_proj']))
        f = tensor.nnet.sigmoid(_slice(preact, 1, options['dim_proj']))
        o = tensor.nnet.sigmoid(_slice(preact, 2, options['dim_proj']))
        c = tensor.tanh(_slice(preact, 3, options['dim_proj']))

        c = f * c_ + i * c
        #c = f * c_printed_ + i * c
        #c = m_[:, None] * c + (1. - m_)[:, None] * c_  # if data is valid m_ = 1, else m_ = 0 (ignore the data)
        #c_printed = theano.printing.Print('c in step')(c)
        #c = c_printed 

        h = o * tensor.tanh(c)
        #h = m_[:, None] * h + (1. - m_)[:, None] * h_
        #h_printed = theano.printing.Print('h in step')(h)
        #h = h_printed

        if options['use_dropout']:
            #h = dropout_layer(h, use_noise, trng)
            proj = h * 0.5
        else: 
            proj = h
        
        #proj_printed = theano.printing.Print('proj in step')(proj)

        y = tensor.dot(proj, tparams['U']) + tparams['b']  # tparams['U'] has size of dim_proj x output_dim (128 x 1 for sinewave example)
        #y = tensor.dot(proj_printed, tparams['U']) + tparams['b']  # tparams['U'] has size of dim_proj x output_dim (128 x 1 for sinewave example)
        #y_printed = theano.printing.Print('y in step')(y) 
        #y = y_printed
        #print "h.ndim: ", h.ndim
        #print "c.ndim: ", c.ndim
        #print "y.ndim: ", y.ndim
        return h, c, y

    h0 = tensor.matrix('h', dtype=config.floatX) # 1 x n_samples x dim_proj
    c0 = tensor.matrix('c', dtype=config.floatX) # 1 x n_samples x dim_proj
    y0 = tensor.tensor3('y', dtype=config.floatX) # input_dim x n_samples x output_dim

    #n_timesteps = 150
    #input_dim = y0.shape[0]
    n_samples = y0.shape[1]
    output_dim = y0.shape[2]
    dim_proj = options['dim_proj']

    y_taps = []
    for t in xrange(-input_dim, 0, 1):
        y_taps.append(t)

    if input_dim == 1: 
        rval, updates = theano.scan(_step,
                                    #sequences=[mask, state_below],
                                    outputs_info=[dict(initial=h0, taps=[-1]), # h0.ndim = 2 will be preserved
                                                  dict(initial=c0, taps=[-1]), # h0.ndim = 2 will be preserved
                                                  dict(initial=y0.reshape([n_samples, output_dim]), taps=y_taps)], # y0.ndim = 3 will become y_[0].ndim = 2
                                                                                  # usage of taps is tricky!! fuck!!
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_timesteps)

    else: 
        rval, updates = theano.scan(_step,
                                    #sequences=[mask, state_below],
                                    outputs_info=[dict(initial=h0, taps=[-1]), # h0.ndim = 2 will be preserved 
                                                  dict(initial=c0, taps=[-1]), # h0.ndim = 2 will be preserved
                                                  dict(initial=y0, taps=y_taps)], # y0.ndim = 3 will become y_[0].ndim = 2 
                                                                                  # usage of taps is tricky!! fuck!!
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_timesteps)

    f_predpred = theano.function(inputs=[h0, c0, y0], 
                             outputs=[rval[2]], 
                             name='f_pred')
    # rval[0] = h, rval[1] = c, rval[2] = y. 
    # prediction requires y.

    return f_predpred

def build_rnn_pred_model(tparams, options, n_timesteps=1, input_dim=1):
    trng = RandomStreams(1234)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))
    use_noise.set_value(0.)

    prefix=options['encoder']

    #print "part1"

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    #def _step(m_, x_, h_, c_): # m_: mask, x_: input, h_: previous hidden, c_: previous hidden
    def _step(h_, *y_): # m_: mask, h_: previous hidden, *y_: raw inputs
        #print "h_.ndim: ", h_.ndim
        #print "len(y_): ", len(y_)
        #print "y_[0].shape: ", y_[0].shape # y_[0] has size of n_samples x output_dim (output_dim = 1 for sinewave example)
        #print "y_[0].ndim: ", y_[0].ndim

        # build x_ from y_
        x_ = y_[0]
        for y_tmp in y_[1:]:
            x_ = tensor.concatenate([x_, y_tmp], axis=1)

        #print theano.pp(x_)
        #h_printed_ = theano.printing.Print('h_ in step')(h_)
        #x_printed_ = theano.printing.Print('x_ in step')(x_)

        #print "x_.ndim: ", x_.ndim
        # embedding and activation
        emb_ = tensor.dot(x_, tparams['Wemb'])
        #emb_ = tensor.dot(x_printed_, tparams['Wemb']) 
        #print "emb_.ndim: ", emb_.ndim

        #emb_printed_ = theano.printing.Print('emb_ in step')(emb_)

        h = tensor.dot(h_, tparams[_p(prefix, 'U')])
        #h = tensor.dot(h_printed_, tparams[_p(prefix, 'U')])
        h += emb_ # x_
        #h += emb_printed_ # x_
        h += tparams[_p(prefix, 'b')]
        h = tensor.tanh(h)

        if options['use_dropout']:
            #h = dropout_layer(h, use_noise, trng)
            proj = h * 0.5
        else: 
            proj = h
        
        #proj_printed = theano.printing.Print('proj in step')(proj)

        y = tensor.dot(proj, tparams['U']) + tparams['b']  # tparams['U'] has size of dim_proj x output_dim (128 x 1 for sinewave example)
        #y = tensor.dot(proj_printed, tparams['U']) + tparams['b']  # tparams['U'] has size of dim_proj x output_dim (128 x 1 for sinewave example)
        #y_printed = theano.printing.Print('y in step')(y) 
        #y = y_printed
        #print "h.ndim: ", h.ndim
        #print "y.ndim: ", y.ndim
        return h, y

    h0 = tensor.matrix('h', dtype=config.floatX) # 1 x n_samples x dim_proj
    y0 = tensor.tensor3('y', dtype=config.floatX) # input_dim x n_samples x output_dim

    #n_timesteps = 150
    #input_dim = y0.shape[0]
    n_samples = y0.shape[1]
    output_dim = y0.shape[2]
    dim_proj = options['dim_proj']

    y_taps = []
    for t in xrange(-input_dim, 0, 1):
        y_taps.append(t)


    if input_dim == 1:
        rval, updates = theano.scan(_step,
                                    #sequences=[mask, state_below],
                                    outputs_info=[dict(initial=h0, taps=[-1]), # h0.ndim = 2 will be preserved
                                                  dict(initial=y0.reshape([n_samples, output_dim]), taps=y_taps)], # y0.ndim = 3 will become y_[0].ndim = 2
                                                                                  # usage of taps is tricky!! fuck!!
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_timesteps)

    else:
        rval, updates = theano.scan(_step,
                                    #sequences=[mask, state_below],
                                    outputs_info=[dict(initial=h0, taps=[-1]), # h0.ndim = 2 will be preserved
                                                  dict(initial=y0, taps=y_taps)], # y0.ndim = 3 will become y_[0].ndim = 2
                                                                                  # usage of taps is tricky!! fuck!!
                                    name=_p(prefix, '_layers'),
                                    n_steps=n_timesteps)

    f_predpred = theano.function(inputs=[h0, y0], 
                             outputs=[rval[1]], 
                             name='f_pred')
    # rval[0] = h, rval[1] = y. 
    # prediction requires y.

    return f_predpred


def train_lstm(
    dim_proj=128,  # word embeding dimension and LSTM number of hidden units.
    patience=10,  # Number of epoch to wait before early stop if no progress
    max_epochs=5000,  # The maximum number of epoch to run
    dispFreq=10,  # Display to stdout the training progress every N updates
    decay_c=0.,  # Weight decay for the classifier applied to the U weights.
    lrate=0.001, #0.0001,  # Learning rate for sgd (not used for adadelta and rmsprop)
    input_dim = 1, # input_dim, equivalent to Vocabulary size for sentiment analysis
    output_dim = 1, # output_dim, 
    #n_words=10000,  # Vocabulary size
    initializer=ortho_weight, # weight initialization method
    optimizer=adadelta,  # sgd, adadelta and rmsprop available, sgd very hard to use, not recommanded (probably need momentum and decaying learning rate).
    encoder='lstm',  # TODO: can be removed must be lstm.
    saveto='lstm_model.npz',  # The best model will be saved there
    validFreq=370,  # Compute the validation error after this number of update.
    saveFreq=1110,  # Save the parameters after every saveFreq updates
    maxlen=None,  # Sequence longer then this get ignored
    batch_size=16,  # The batch size during training.
    valid_batch_size=64,  # The batch size used for validation/test set.
    dataset='sinewave',

    # Parameter for extra option
    noise_std=0.,
    use_dropout=True,  # if False slightly faster, but worst test error
                       # This frequently need a bigger model.
    reload_model="",  # Path to a saved model we want to start from.
    test_size=-1,  # If >0, we keep only this number of test example.
):

    # Model options
    model_options = locals().copy()
    print "model options", model_options

    load_data, prepare_data = get_dataset(dataset) # load functions to get dataset

    print 'Loading data'
    train, valid, test = load_data(valid_portion=0.05,
                                   maxlen=maxlen)

    ## print minibatch data 
    x, mask, y = prepare_data([train[0][t] for t in [0]], 
                              [train[1][t] for t in [0]], 
                              input_dim, output_dim)
    #print "x.shape: ", x.shape
    #print "y.shape: ", y.shape
    plt.close('all')
    fig_data = plt.figure()
    input_seq = plt.plot(numpy.linspace(1, x.shape[0], x.shape[0]), x.reshape([x.shape[0], x.shape[2]]), label='u_t, input seq')
    output = plt.plot(numpy.linspace(1, x.shape[0], y.shape[0]), y.reshape([y.shape[0], y.shape[2]]), linestyle='--', label='y_t, output')
    plt.title('sample input seq(u_t) and its output(y_t)')
    plt.legend()
    plt.savefig('results/prepare_data_%s_%s_%s_input_dim_%d.png' % (model_options['encoder'], optimizer.__name__, initializer.__name__, input_dim))

    ydim = output_dim #numpy.max(train[1]) + 1 # why should y dimension be one larger than existing y values? 
    #print 'ydim %d' % numpy.max(train[1])
    #print train 
    #print valid 
    #print test
    #print len(train[0])
    #print len(valid[0])
    #print len(test[0])
    #raise NameError('HiThere')

    model_options['ydim'] = ydim

    print 'Building model'
    # This create the initial parameters as numpy ndarrays.
    # Dict name (string) -> numpy ndarray
    params = init_params(model_options)
    # params now has following objects                                            #        params[Wemb], params[lstm_W], params[lstm_U], params[lstm_b],
    #                                      params[U], params[b]
    # params with prefix lstm_ have initialzed with ortho_weight function

    if reload_model:
        #load_params('lstm_model.npz', params)
        load_params(reload_model, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)
    # tparams now has following objects
    #        tparams[Wemb], tparams[lstm_W], tparams[lstm_U], tparams[lstm_b],
    #                                      tparams[U], tparams[b]

    # use_noise is for dropout
    (use_noise, x, mask,
     y, f_pred, cost, f_hiddens) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        weight_decay += (tparams['U'] ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_cost = theano.function([x, mask, y], cost, name='f_cost')
    
    grads = tensor.grad(cost, wrt=tparams.values())
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)
    
    print 'Optimization'

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print "%d train examples" % len(train[0])
    print "%d valid examples" % len(valid[0])
    print "%d test examples" % len(test[0])
    history_errs = []
    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size
    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

 
    uidx = 0  # the number of update done
    estop = False  # early stop
    start_time = time.clock()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0
           
            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)
            #print "batchsize = %d" % batch_size
            #print "len(kf) = %d" % len(kf)
            #print kf 
            #raise NameError('HiThere')
            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                #print x 
                #print y 
                #print len(x)
                #print len(y)
                #print len(x[0])
                x, mask, y = prepare_data(x, y, input_dim, output_dim)
                #print "x.shape", x.shape 
                #print "y.shape", y.shape
                #print "len(y) = %d" % (len(y))
                #print x 
                #print y 
                #print "asdf"
                n_samples += x.shape[1]
             
                #cost_temp = f_cost(x, mask, y)
                #print cost_temp
                #asdf = f_pred(x, mask)
                #print asdf
                #print "asdf.shape: ", asdf.shape
                    
                cost = f_grad_shared(x, mask, y)
                f_update(lrate)
  
                #print "kasd;adfs;j;fadsk;"
                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost

                if saveto and numpy.mod(uidx, saveFreq) == 0:
                    print 'Saving...',

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'), -1)
                    print 'Done'

                if numpy.mod(uidx, validFreq) == 0:
                    use_noise.set_value(0.)
                    train_err = pred_error(f_pred, prepare_data, input_dim, output_dim, train, kf)
                    valid_err = pred_error(f_pred, prepare_data, input_dim, output_dim, valid,
                                           kf_valid)
                    test_err = pred_error(f_pred, prepare_data, input_dim, output_dim, test, kf_test)

                    history_errs.append([valid_err, test_err])

                    if (uidx == 0 or
                        valid_err <= numpy.array(history_errs)[:,
                                                               0].min()):

                        best_p = unzip(tparams)
                        bad_counter = 0

                    print ('Train ', train_err, 'Valid ', valid_err,
                           'Test ', test_err)

                    if (len(history_errs) > patience and
                        valid_err >= numpy.array(history_errs)[:-patience,
                                                               0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

            print 'Seen %d samples' % n_samples

            if estop:
                break

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.clock()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, input_dim, output_dim, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, input_dim, output_dim, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, input_dim, output_dim, test, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    if max_epochs > 0: 
        if saveto:
            numpy.savez(saveto, train_err=train_err,
                        valid_err=valid_err, test_err=test_err,
                        history_errs=history_errs, **best_p)
        print 'The code run for %d epochs, with %f sec/epochs' % (
            (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
        print >> sys.stderr, ('Training took %.1fs' %
                              (end_time - start_time))


    ##################################### prediction
    print 'Prediction' 
    use_noise.set_value(0.)

    if model_options['encoder']=='lstm':
        f_predpred = build_lstm_pred_model(tparams, model_options, n_timesteps=500, input_dim=input_dim)
    elif model_options['encoder']=='rnn':
        f_predpred = build_rnn_pred_model(tparams, model_options, n_timesteps=500, input_dim=input_dim)
 
    #kf = get_minibatches_idx(len(test[0]), batch_size, shuffle=True)
    kf = get_minibatches_idx(len(test[0]), 5, shuffle=True)
    #print "batchsize = %d" % batch_size
    #print "len(kf) = %d" % len(kf)
    #print [kf[0]]
    #raise NameError('HiThere')
    for _, test_index in [kf[0]]:

        # Select the random examples for this minibatch
        y = [test[1][t] for t in test_index] # batchsize number of seqs
        x = [test[0][t] for t in test_index] # 
#        y = [test[1][0]] # single seq
#        x = [test[0][0]] # single seq
        x_true = x

        for t in xrange(len(test_index)):
            # We just plot one of the sequences
            plt.close('all')
            fig_pred = plt.figure()

            # Graph 1
            input_seq, = plt.plot(x[t], label='x_t, input seq')
            true_targets, = plt.plot(y[t], label='y_t, true ouput')
            plt.grid()
            plt.legend()
            plt.title('sample test data, test_index =' % test_index[t])

            # Save as a file
            plt.savefig('results/data_%s_%s_%s_input_dim_%d_test_index_%d.png' % (model_options['encoder'], optimizer.__name__, initializer.__name__, input_dim, test_index[t]))

 
        x, mask, y = prepare_data(x, y, input_dim, output_dim)
        xx = x[:50, :, :]
        maskmask = mask[:50, :]
        yy = y[:50, :]
  
        preds_y_all = f_pred(x, mask) # for debugging
        
        preds_y = f_pred(xx, maskmask)
        if model_options['encoder']=='lstm':
            preds_h = f_hiddens[0](xx, mask)
            preds_c = f_hiddens[1](xx, maskmask)

            predpred = f_predpred(preds_h[-1], preds_c[-1], preds_y[-input_dim:])
            #predpred = f_predpred(preds_h[-1], preds_c[-1], x[50,:,:].reshape([input_dim, 1, 1]))
        elif model_options['encoder']=='rnn':
            preds_h = f_hiddens[0](xx, mask)
            predpred = f_predpred(preds_h[-1], preds_y[-input_dim:])
            #predpred = f_predpred(preds_h[-1], x[50,:,:].reshape([input_dim, 1, 1]))
          
        #print "preds_y.shape: ", preds_y.shape
        #print "len(predpred): ", len(predpred)
        #print "predpred[0].shape", predpred[0].shape

        y_pred_total = numpy.concatenate((preds_y, predpred[0]), axis=0)
    
        #print "y_pred_total: ", y_pred_total.shape

        for t in xrange(len(test_index)):
            # We just plot one of the sequences
            plt.close('all')
            fig_pred = plt.figure()

            # Graph 1
            #input_seq, = plt.plot(x[:,t,-1], label='x_t, input seq')
            true_targets, = plt.plot(y[:,t], label='y_t, true ouput')
            #ggg, = plt.plot(preds_y_all[:,t,0], linestyle='--', label='y_t_pred, model output')
            guessed_targets, = plt.plot(y_pred_total[:,t,0], linestyle='--', linewidth=2, label='y_t_pred, model output')
            verticalline = plt.axvline(x=49, color='r', linestyle=':')
            plt.grid()
            plt.legend()
            plt.title('true output vs. model output, test_index = ' % test_index[t])
            plt.ylim((-1.5, 1.5))
  
            # Save as a file
            plt.savefig('results/pred_%s_%s_%s_input_dim_%d_test_index_%d.png' % (model_options['encoder'], optimizer.__name__, initializer.__name__, input_dim, test_index[t]))

    
    return train_err, valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    input_dim = 5
    dim_proj = 10
    encoder = 'lstm'
    #mode = 'train'
    mode = 'test'
    optimizer=adadelta
    initializer=uniform_weight
    filename="models/model_%s_%s_%s_%d.npz" % (encoder, optimizer.__name__, initializer.__name__, input_dim)
    if mode=='train':
        train_lstm(
            dim_proj=dim_proj, 
            input_dim=input_dim, 
            #reload_model=filename,
            saveto=filename, 
            max_epochs=400,
            test_size=500,
            use_dropout=False,
            encoder=encoder,
            initializer=initializer,
            optimizer=optimizer
        )
    else: 
        train_lstm(
            dim_proj=dim_proj,
            input_dim=input_dim, 
            reload_model=filename,
            #saveto=filename, 
            max_epochs=0,
            test_size=500,
            use_dropout=False,
            encoder=encoder,
            initializer=initializer, 
            optimizer=optimizer
        )
