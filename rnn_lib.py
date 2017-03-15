from __future__ import division
import numpy as np
import theano
import lasagne

### BE CAREFUL WITH DIVISION IN PYTHON.2-3 ###

def init_normalized(n):
    A = np.array(np.random.normal(size=(n, n)),dtype = 'float32')
    A = np.triu(A, k = 1) + np.triu(A).T
    eig, _ = np.linalg.eig(A + np.eye(n,dtype='float32'))
    e = max(eig)
    W = (A + np.eye(n,dtype='float32')) / e
    return W
    
def init_posdef_w(n):
    # A = 1/N R^T * R, wehere R is standard normal
    # A is positive definite
    # W = (A + I) / max(spec(A + I))
    # spec(W) = {1, l_i < 1}
    
    R = np.array(np.random.normal(size=(n, n)),dtype = 'float32')
    A = np.dot(R.T, R) / n
    eig, _ = np.linalg.eig(A + np.eye(n,dtype='float32'))
    e = max(eig)
    W = (A + np.eye(n,dtype='float32')) / e
    return W

def build_network(inp, seq_len, dim, nhidden, nout, nonlinear, rnn_type):
    l_in = lasagne.layers.InputLayer(shape=(None, seq_len, dim), input_var=inp)
    alpha = np.sqrt(2) * np.exp(1.2 / (max(nhidden, 6) - 2.4))
    hid_to_hid = None
    
    if rnn_type == 'srnn':
        hid_to_hid = lasagne.init.Normal()
    elif rnn_type == 'nrnn':
        hid_to_hid = init_normalized(nhidden)
    elif rnn_type == 'irnn':
        hid_to_hid = np.eye(nhidden, dtype='float32')
    elif rnn_type == 'nprnn':
        hid_to_hid = init_posdef_w(nhidden)
    else:
        print("RNN type unknown", rnn_type)
        raise
    
    l_rnn = lasagne.layers.RecurrentLayer(
        l_in, nhidden,
        W_in_to_hid=lasagne.init.Normal(std=alpha / np.sqrt(nhidden), mean=0.0),
        W_hid_to_hid=hid_to_hid,
        learn_init=True,
        only_return_final=True,
        nonlinearity=lasagne.nonlinearities.rectify)

    l_out = lasagne.layers.DenseLayer(l_rnn, num_units=nout,
                                      W=lasagne.init.GlorotNormal(),
                                      nonlinearity=nonlinear)
    return l_out
