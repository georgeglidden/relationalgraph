from model import *
import augmentations
import numpy as np
from itertools import chain
from skimage.io import imread,imshow,show
import tensorflow as tf

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

def check_samples(samples,labels,T,lb):
    assert len(samples) == len(labels)
    e = 0
    for i in range(len(samples)):
        x = samples[i].numpy()
        y = labels[i].numpy()
        if not T(x, y):
            e += 1
            print(f'{lb} sample: {(x,y)}')
    return e, e/len(samples)

def check_aggregate():
    is_pos_tuple = lambda x, y: x[0] == x[1]
    is_neg_tuple = lambda x, y: x[0] != x[1]
    K = 10
    M = 10
    P = M*((K-1)**2)
    mset = np.random.uniform(0,M,M).astype(np.int64)
    X = list(chain.from_iterable([[m for k in range(K)] for m in mset]))
    samples_pos, labels_pos = aggregate_pos(X, X, M, K)
    ep = check_samples(samples_pos, labels_pos, is_pos_tuple, 'pos')
    print('pos agg results')
    if len(samples_pos) == P:
        print('PASSED:')
    else:
        print('FAILED:')
    print('\texpected nb samples {}'.format(P))
    print('\treal nb samples {}'.format(len(samples_pos)))
    if ep[0] == 0:
        print('PASSED:')
    else:
        print('FAILED:')
    print('\tpositive sample error, error rate {}, {}'.format(ep[0], ep[1]))

    samples_neg, labels_neg = aggregate_neg(X,X,M,K)
    en = check_samples(samples_neg, labels_neg, is_neg_tuple, 'neg')
    print('neg agg results')
    if len(samples_neg) == P:
        print('PASSED:')
    else:
        print('FAILED:')
    print('\texpected nb samples {}'.format(P))
    print('\treal nb samples {}'.format(len(samples_neg)))
    if en[0] == 0:
        print('PASSED:')
    else:
        print('FAILED:')
    print('\tnegative sample error, error rate {}, {}'.format(en[0], en[1]))

    samples, labels = aggregate(X,X,M,K)
    def well_formed(s, l):
        if (s[0] == s[1] and l == 1) or (s[0] != s[1] and l == 0):
            return True
        else:
            return False
    assert well_formed([1,1],1) == is_pos_tuple([1,1],1)
    assert well_formed([0,1],1) == is_pos_tuple([0,1],1)
    assert well_formed([0,1],0) == is_neg_tuple([0,1],0)
    assert well_formed([1,1],0) == is_neg_tuple([1,1],0)
    e = check_samples(samples, labels, well_formed, 'any')
    print('agg results')
    if len(samples) == 2*P:
        print('PASSED:')
    else:
        print('FAILED:')
    print('\texpected nb samples {}'.format(2*P))
    print('\treal nb samples {}'.format(len(samples)))
    if e[0] == 0:
        print('PASSED:')
    else:
        print('FAILED:')
    print('\tnb sample error, error rate {}, {}'.format(e[0], e[1]))
    print(mset)

def check_augmentations():
    aug = [augmentations.apply_all] + augmentations.all_fns
    N = 4
    I = len(aug)
    T = 5
    for n in range(0,N):
        path = f'sample0{n+1}.jpeg'
        im = imread(path, as_gray=False)
        w,h,c = im.shape
        for i in range(I):
            print(aug[i])
            for t in range(T):
                imshow(aug[i](im))
                show()

def check_batch_to_agg():
    (x,y),(_,_) = tf.keras.datasets.cifar10.load_data()
    N = x.shape[0]
    M = 3
    K = 10
    batches = batch(x,y,N,M,K)
    for mb in batches:
        bx, by = mb()
        print('batch')
        print(bx.shape)
        pos_samples, pos_labels = aggregate_pos(bx,by,M,K)
        neg_samples, neg_labels = aggregate_neg(bx,by,M,K)
        print(pos_samples.shape)
        print(neg_samples.shape)
        samples, labels = aggregate(bx, by, M, K)

#check_batch()
#check_aggregate()
#check_augmentations()
check_batch_to_agg()
