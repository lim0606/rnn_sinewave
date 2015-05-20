import cPickle
import gzip
import os

import numpy
import theano
import math

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ion()

def prepare_data(seqs, labels, input_dim, stride=1, output_dim=None, maxlen=None):

    if output_dim is None:
        output_dim=input_dim

    set_dim=input_dim+output_dim

    #print seqs
    #print labels
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)
    #print "maxlen = %d" % maxlen 
    #print "len(seqs) = %d, len(labels) = %d" % (len(seqs), len(labels))
    x = numpy.zeros(( (maxlen-set_dim)/stride+1, n_samples, input_dim)).astype(theano.config.floatX)
    x_mask = numpy.zeros(( (maxlen-set_dim)/stride+1, n_samples)).astype(theano.config.floatX)
    y = numpy.zeros(( (maxlen-set_dim)/stride+1, n_samples, output_dim)).astype(theano.config.floatX)

    for idx, (s, l) in enumerate(zip(seqs, labels)):
        for i in range(0, (lengths[idx]-set_dim)/stride+1):
            x[i, idx, :] = s[i*stride:i*stride+input_dim]
            y[i, idx, :] = s[i*stride+input_dim:i*stride+input_dim+output_dim]

        #print "input_dim = %d" % input_dim 
        #print "length = %d" % lengths[idx]
        x_mask[:(lengths[idx]-set_dim)/stride+1, idx] = 1.

    #print "x.shape: ", x.shape
    #print "y.shape: ", y.shape
    return x, x_mask, y
    # x = n_timestamp x n_sample x input_dim
    # x_mask = n_timestamp x n_sample
    # y = n_timestamp x n_sample
    #    where n_timestamp = maxlen - input_dim + 1


def load_data(path="sinewave.pkl", valid_portion=0.1, maxlen=None, sort_by_len=False):

    '''Loads the dataset

    :type valid_portion: float
    :param valid_portion: The proportion of the full train set used for
        the validation set.
    :type maxlen: None or positive int
    :param maxlen: the max sequence length we use in the train/valid set.
    :type sort_by_len: bool
    :name sort_by_len: Sort by the sequence lenght for the train,
        valid and test set. This allow faster execution as it cause
        less padding per minibatch. Another mechanism must be used to
        shuffle the train set at each epoch.

    '''

    #############
    # LOAD DATA #
    #############

    gen_data = False

    # Load the dataset
    try:
        f = open(path, 'rb')
    except:
        gen_data = True
        f = open(path, 'wb')

    if gen_data is True:
        #############
        # GEN DATA  #
        #############    
        print 'Generate sine wave dataset'
        n_sinewave = 1000 #20000
        n_train = n_sinewave # number of sine wave sequences (train)
        n_test = n_sinewave # number of sine wave sequences (test)
        length_sinewave = 160 # length of each sine wave sequence

        seq_sine_alpha_train = numpy.random.uniform(1, 5, (n_train, 1))
        seq_sine_alpha_test = numpy.random.uniform(1, 5, (n_test, 1))
        seq_sine_beta_train = numpy.random.uniform(-1, 1, (n_train, 1))
        seq_sine_beta_test = numpy.random.uniform(-1, 1, (n_test, 1))

        #print seq_sine_alpha_train.shape
        #print seq_sine_beta_train.shape

        '''sinewave_base = numpy.linspace(0, 2 * math.pi, length_sinewave)
        sinewave_x = sinewave_base[:-1:]
        sinewave_xp1 = sinewave_base[1::]
        #print "len(sinewave_x) = %d, len(sinewave_xp1) = %d" %(len(sinewave_x), len(sinewave_xp1))
        '''
        sinewave_x = numpy.linspace(0, 2 * math.pi, length_sinewave)

        train_set_x = []
        train_set_y = []
        test_set_x = []
        test_set_y = []

        for i in range(n_train):
            train_set_x.append(numpy.sin(seq_sine_alpha_train[i] * sinewave_x + seq_sine_beta_train[i] * 2 * math.pi).tolist())
            #train_set_y.append(numpy.sin(seq_sine_alpha_train[i] * sinewave_xp1 + seq_sine_beta_train[i] * 2 * math.pi).tolist())
            train_set_y.append( ( -1 * numpy.ones(length_sinewave) ).tolist() )

        train_set = (train_set_x, train_set_y)
        #print "train_set: " 
        #print train_set
        del train_set_x, train_set_y

        for i in range(n_test):
            test_set_x.append(numpy.sin(seq_sine_alpha_test[i] * sinewave_x + seq_sine_beta_test[i] * 2 * math.pi).tolist())
            #test_set_y.append(numpy.sin(seq_sine_alpha_test[i] * sinewave_xp1 + seq_sine_beta_test[i] * 2 * math.pi).tolist())
            test_set_y.append( ( -1 * numpy.ones(length_sinewave) ).tolist() )

        test_set = (test_set_x, test_set_y)    
        del test_set_x, test_set_y

        #print train_set == test_set
        #print train_set[0] == test_set[0]
        #print train_set[0] # a list with length 25000
        #print train_set[0][0] # a list with sentence's length (can vary)
        #print test_set[0][0]
        #print f
        #print "len(train_set) = %d" % len(train_set)
        #print "len(train_set[0]) = %d, len(train_set[1]) = %d" % (len(train_set[0]), len(train_set[1]))
        #print "len(train_set[0][0]) = %d, len(train_set[1][0]) = %d" % (len(train_set[0][0]), len(train_set[1][0])) 
        #print maxlen 
        if maxlen:
            new_train_set_x = []
            new_train_set_y = []
            for x, y in zip(train_set[0], train_set[1]):
                if len(x) <= maxlen:
                    new_train_set_x.append(x)
                    new_train_set_y.append(y)
            train_set = (new_train_set_x, new_train_set_y)
            #print "train_set: "
            #print train_set
            del new_train_set_x, new_train_set_y

        # split training set into validation set
        train_set_x, train_set_y = train_set
        #print "train_set: "
        #print train_set
        n_samples = len(train_set_x)
        sidx = numpy.random.permutation(n_samples)
        n_train = int(numpy.round(n_samples * (1. - valid_portion)))
        valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
        valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
        train_set_x = [train_set_x[s] for s in sidx[:n_train]]
        train_set_y = [train_set_y[s] for s in sidx[:n_train]]

        train_set = (train_set_x, train_set_y)
        valid_set = (valid_set_x, valid_set_y)


        cPickle.dump(train_set, f, -1)
        cPickle.dump(valid_set, f, -1)
        cPickle.dump(test_set, f, -1)   
    else:
        train_set = cPickle.load(f) # tuple with lenth 2
        valid_set = cPickle.load(f) 
        test_set = cPickle.load(f)
 
    '''
    plt.close('all')
    fig = plt.figure()
    print "len(sinewave_base) = %d " % len(sinewave_base)
    print "len(train_set[0]) = %d " % len(train_set[0][0])
    input_seq1 = plt.plot(sinewave_base[:-1], train_set[0][0], label='input seq1')
    input_seq2 = plt.plot(sinewave_base[:-1], test_set[0][0], label='input seq2')
    plt.title('sample input sequences')
    plt.legend()
    plt.savefig('data_lstm_input_dim_5.png')'''
 
    #print "train_set: "
    #print train_set
    return train_set, valid_set, test_set  
    # train_set, valid_set, test_set is tuple with two element x and y
    # Each tuple element is a list with length its number of data
