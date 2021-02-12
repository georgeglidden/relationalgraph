from random import randint
import tensorflow as tf

def aggregate_pos(X, M, K):
    aggX = list()
    aggT = list()
    # M(K^2-K) positive tuples
    for m in range(M):
        s = m*K
        for i in range(K):
            x1 = X[s+i]
            for j in range(i+1,K):
                if i==j:
                    continue
                x2 = X[s+j]
                aggX.append(tf.concat((x1,x2),0))
                aggT.append(1.0)
    return tf.stack(aggX), tf.stack(aggT)

def aggregate_neg(X, M, K):
    aggX = list()
    aggT = list()
    # M(K^2-K) negative tuples
    for m in range(M):
        s = m*K
        for i in range(K):
            x1 = X[s+i]
            for j in range(K):
                n = m
                while n == m:
                    n = randint(0,M-1)
                t = n*K
                x2 = X[t+j]
                aggX.append(tf.concat((x1,x2),0))
                aggT.append(0.0)
    return tf.stack(aggX), tf.stack(aggT)

# X is a set of augmented data containing M sets of K augmentations
# so the first K items are of the same class, the second K are of a
# different class, etc.
def pair(X, M, K):
    assert len(X) == M * K
    pos_s, pos_t = aggregate_pos(X,M,K)
    neg_s, neg_t = aggregate_neg(X,M,K)
    if neg_s.shape[0] == 0:
        return pos_s, pos_t
    agg_s = tf.concat((pos_s,neg_s),0)
    agg_t = tf.concat((pos_t,neg_t),0)
    permutation = tf.random.shuffle(tf.range(tf.shape(agg_s)[0]))
    return tf.gather(agg_s, permutation, axis=0), tf.gather(agg_t, permutation, axis=0)
