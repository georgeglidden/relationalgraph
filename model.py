import tensorflow as tf
import numpy as np
import augmentations

rng = np.random.default_rng()

# reduce memory used by augmented datasets
class MiniBatch():
    def __init__(self, samples, labels, M, K, augmentor):
        self._source = (samples, labels)
        self._M = M
        self._K = K
        self._augmentor = augmentor

    # returns a list of K*M items, the first K {0, ..., K-1} of which are augmentations on the first source item, the second K {K, ..., 2K-1} augmentations of the second, etc., where K = nb_augment and M = nb_source.
    def __call__(self):
        batch_x = list()
        batch_y = list()
        for m in range(self._M):
            x = self._source[0][m]
            y = self._source[1][m]
            w,h = x.shape[:2]
            for k in range(self._K):
                assert not np.any(np.isnan(x))
                aug_x = tf.reshape(self._augmentor(x), (w,h,-1))
                assert not np.any(np.isnan(aug_x))
                batch_x.append(aug_x)
                batch_y.append(y)
        batch_tensor_x, batch_tensor_y = tf.stack(batch_x), tf.stack(batch_y)
        return batch_tensor_x, batch_y

# generate ceil(N/M) MiniBatches from ds
#   ds      :   source dataset, iterable over (x, label) tuples
#   N       :   tuples in dataset
#   M       :   batch size
#   K       :   number of augmentations
# augmentor :   augment fn
# ds, N, M, K -> [MiniBatch(ds[i:min(i+M,N)], M, apply_random_augmentations, K) for
#                   i in range(0,N,M)]
def batch(dx, dy, N, M, K, augmentor = lambda x: x, shuffle=True):
    batched_ds = list()
    for i in range(0, N, M):
        j = min(i+M, N)
        minibatch = MiniBatch(dx[i:j], dy[i:j], M, K, augmentor)
        batched_ds.append(minibatch)
    if shuffle:
        np.random.shuffle(batched_ds)
    return batched_ds

def aggregate_pos(X, Y, M, K):
    aggX = list()
    aggT = list()
    # M(K^2-K) positive tuples
    for m in range(M):
        for k in range(K):
            x1 = X[m*K + k]
            for j in range(K):
                if Y[m*K + k] != Y[m*K + j]:
                    # screen out negative tuples
                    continue
                x2 = X[m*K + j]
                x = tf.concat((x1,x2), 0)
                aggX.append(x)
                aggT.append(1.0)
    return tf.stack(aggX), tf.stack(aggT)

def aggregate_neg(X, Y, M, K):
    aggX = list()
    aggT = list()
    # M(K^2-K) negative tuples
    for m in range(M):
        for k in range(K):
            x1 = X[m*K + k]
            for i in range(K-1):
                neg_m = (m + int(np.ceil(rng.random()*(M-1)))) % M
                neg_k = int(rng.random())*K
                if Y[m*K + k] == Y[neg_m*K + neg_k]:
                    # screen out positive tuples
                    continue
                x2 = X[neg_m*K + neg_k]
                x = tf.concat((x1,x2), 0)
                aggX.append(x)
                aggT.append(0.0)
    return tf.stack(aggX), tf.stack(aggT)

def aggregate(X, Y, M, K):
    pos_s, pos_t = aggregate_pos(X,Y,M,K)
    neg_s, neg_t = aggregate_neg(X,Y,M,K)
    if neg_s.shape[0] == 0:
        return pos_s, pos_t
    agg_s = tf.concat((pos_s,neg_s),0)
    agg_t = tf.concat((pos_t,neg_t),0)
    return agg_s, agg_t

class Conv4(tf.keras.Model):
    def __init__(self):
        super(Conv4, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, 3, name='bb_conv1')
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.relu1 = tf.keras.layers.ReLU()
        self.pool1 = tf.keras.layers.AvgPool2D(strides=2)

        self.conv2 = tf.keras.layers.Conv2D(16, 3, name='bb_conv2')
        self.norm2 = tf.keras.layers.BatchNormalization()
        self.relu2 = tf.keras.layers.ReLU()
        self.pool2 = tf.keras.layers.AvgPool2D(strides=2)

        self.conv3 = tf.keras.layers.Conv2D(32, 3, name='bb_conv3')
        self.norm3 = tf.keras.layers.BatchNormalization()
        self.relu3 = tf.keras.layers.ReLU()
        self.pool3 = tf.keras.layers.AvgPool2D(strides=2)

        self.flatten4 = tf.keras.layers.Flatten()
        self.dense4 = tf.keras.layers.Dense(64, name='bb_dense4')
        self.norm4 = tf.keras.layers.BatchNormalization()
        self.relu4 = tf.keras.layers.ReLU()

    def call(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten4(x)
        x = self.dense4(x)
        x = self.norm4(x)
        x = self.relu4(x)

        return x

class Relator(tf.keras.Model):
    def __init__(self):
        super(Relator, self).__init__()
        self.d1 = tf.keras.layers.Dense(256, name='rel_dense1')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.lru = tf.keras.layers.LeakyReLU()
        self.d2 = tf.keras.layers.Dense(1,activation='softmax', name='rel_dense2')

    def call(self, x):
        x = self.d1(x)
        x = self.bn1(x)
        x = self.lru(x)
        x = self.d2(x)
        return x

@tf.function
def loss(T, Y):
    return loss_object(T,Y)

def train_step(X, T1, M, K, step):
    with tf.GradientTape(persistent=True) as tape:
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(T1))
        # encoding
        Z1 = backbone(X, training=True)
        assert not np.any(np.isnan(Z1))
        # aggregation
        Z2, T2 = aggregate(Z1, T1, M, K)
        assert not np.any(np.isnan(T2))
        assert not np.any(np.isnan(Z2))
        # score tuples
        Y = relation(Z2, training=True)
        # approximate crossentropy
        assert not np.any(np.isnan(Y))
        model_loss = loss(T2, Y)
    relation_gradients = tape.gradient(model_loss, relation.trainable_variables)
    optimizer.apply_gradients(zip(relation_gradients, relation.trainable_variables))
    backbone_gradients = tape.gradient(model_loss, backbone.trainable_variables)
    optimizer.apply_gradients(zip(backbone_gradients, backbone.trainable_variables))
    train_loss(T2, Y)
    train_accuracy(T2, Y)

def test_step(X, T1, M, K):
    Z1 = backbone(X, training=False)
    Z2, T2 = aggregate(Z1, T1, M, K)
    Y = relation(Z2, training=False)
    l = loss(T2, Y)
    test_loss(T2, Y)
    test_accuracy(T2, Y)

if not __name__ == '__main__':
    print('model.py')
else:
    #tf.keras.backend.set_floatx('float64')
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.cifar10.load_data()
    train_perm = tf.random.shuffle(tf.range(tf.shape(train_x)[0]))
    train_x = tf.gather(train_x, train_perm, axis=0)
    train_y = tf.gather(train_y, train_perm, axis=0)
    test_perm = tf.random.shuffle(tf.range(tf.shape(test_x)[0]))
    test_x = tf.gather(test_x, test_perm, axis=0)
    test_y = tf.gather(test_y, test_perm, axis=0)
    M = 2
    K = 10
    p = 1000
    rate = 1e-4
    EPOCHS = 20
    N1 = len(train_x)
    N2 = len(test_x)
    backbone = Conv4()
    backbone.build((None,32,32,3))
    backbone.summary()
    relation = Relator()
    relation.build((None,128))
    relation.summary()
    train_loss = tf.keras.metrics.BinaryCrossentropy()
    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    test_loss = tf.keras.metrics.BinaryCrossentropy()
    test_accuracy = tf.keras.metrics.BinaryAccuracy()
    loss_object = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(rate)
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()
        train_batches = batch(train_x, train_y, N1, M, K, augmentor=augmentations.apply_all)
        test_batches = batch(test_x, test_y, N2, M, K, augmentor=augmentations.apply_all)
        print('training')
        for i in range(N1):
            minibatch = train_batches[i]
            train_batch_x, train_batch_y = minibatch()
            train_step(train_batch_x, train_batch_y, M, K, i%2)
            if i % (N1//p) == 0:
                print(
                f'{100*i/N1}% '
                f'{train_loss.result()} ')
        print('testing')
        for i in range(N2):
            minibatch = test_batches[i]
            test_batch_x, test_batch_y = minibatch()
            test_step(test_batch_x, test_batch_y, M, K)
            if i % (N2//p) == 0:
                print(
                f'{100*i/N2}%')
        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )
