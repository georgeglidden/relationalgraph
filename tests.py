from model import *
import numpy as np
from itertools import chain

def check_batch():
    N = 1000
    M = 10
    ds = np.arange(N)

    batched_ds = batch(ds, N, M, K=1, shuffle=False)
    chained_ds = list(chain.from_iterable([mb() for mb in batched_ds]))
    if list(ds) == chained_ds:
        print('PASSED:', end=' ')
    else:
        print('FAILED:', end=' ')
    print('unshuffled batch K=1 maintains list identity')

    batched_ds = batch(ds, N, M, K=5, shuffle=True)
    chained_ds = list(chain.from_iterable([mb() for mb in batched_ds]))
    if set(ds) == set(chained_ds):
        print('PASSED:', end=' ')
    else:
        print('FAILED:', end=' ')
    print('shuffled batch K=5 maintains set identity')
    print('\tshuffled batch contains {} items'.format(len(list(chained_ds))))

def check_samples(samples,labels,T):
    assert len(samples) == len(labels)
    e = 0.0
    for i in range(len(samples)):
        x = samples[i].numpy()
        y = labels[i].numpy()
        if not T(x, y):
            e += 1
    return e / len(samples)

def check_aggregate():
    is_pos_tuple = lambda x, y: x[0]*x[1] > 0
    is_neg_tuple = lambda x, y: x[0]*x[1] < 0
    X = [0.9,1.1,0.86,1.02,-1.3,-0.7,-1.04,-1.1]
    M = 2
    K = 4
    P = M*((K-1)**2)
    samples_pos, labels_pos = aggregate_pos(X,M,K)
    ep = check_samples(samples_pos, labels_pos, is_pos_tuple)
    samples_neg, labels_neg = aggregate_neg(X,M,K)
    en = check_samples(samples_neg, labels_neg, is_neg_tuple)
    print('expected nb samples {}'.format(P))
    print('real nb samples {} and {}'.format(len(samples_pos),len(samples_neg)))
    print('positive sample error rate {}'.format(ep/P))
    print('negative sample error rate {}'.format(en/P))

check_batch()
check_aggregate()
