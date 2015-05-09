import numpy 
import theano
import theano.tensor as T

def fn(h_tm1, *y_tms):
    print "h_tms1: ", h_tm1
    #y_tms_concat = y_tms[0]
    print "y_tms[0]:", y_tms[0]
    print "y_tms[1:]:"
    for y_tm in y_tms[1:]: 
        print y_tm
    return 

print "start"
print "fn(1, 2, 3, 4, 5)"
fn(1, 2, 3, 4, 5)

print "start"
print "fn(1, 2)"
fn(1, 2)

    
def recurrent_fn2(h_tm1, *y_tms):
    y_tms_concat = y_tms[0]
    for y_tm in y_tms[1:]:
        y_tms_concat = T.concatenate([y_tms_concat, y_tm], axis=1)

    h_t = T.tanh(T.dot(y_tms_concat, W_uh) + \
                          T.dot(h_tm1, W_hh) + \
                          b_h)
    y_t = T.dot(h_t, W_hy) + b_y
    return h_t, y_t


def fn2(): 
    return 1, [2, 3]

def fn3():
    return 1, [2]

def fnfn(num):
    if num == 2:
        return fn2()
    elif num == 3:
        return fn3()
    else:
        return 1, None

a, bs = fnfn(1)
if bs is not None:
    for b in bs:
        print b

a, bs = fnfn(2)
if bs is not None:
    for b in bs:
        print b

a, bs = fnfn(3)
if bs is not None:
    for b in bs:
        print b


