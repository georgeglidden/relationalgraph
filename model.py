import tensorflow as tf
import numpy as np
import PIL

rng = np.random.default_rng()

# reduce memory used by augmented datasets
class MiniBatch():
    def __init__(self, source, M, K, augmentor):
        self._source = source
        self._M = M
        self._K = K
        self._augmentor = augmentor

    # returns a list of K*M items, the first K {0, ..., K-1} of which are augmentations on the first source item, the second K {K, ..., 2K-1} augmentations of the second, etc., where K = nb_augment and M = nb_source.
    def __call__(self):
        batch = list()
        for m in range(self._M):
            x = self._source[m]
            for k in range(self._K):
                # TODO: convert augmentation fns to np/tensorflow operations
                #       reduce conversion overhead
                aug_x = self._augmentor(x, k)
                batch.append(aug_x)
        return batch

# generate ceil(N/M) MiniBatches from ds
#   ds      :   source dataset, iterable over (x, label) tuples
#   N       :   tuples in dataset
#   M       :   batch size
#   K       :   number of augmentations
# augmentor :   augment fn
# ds, N, M, K -> [MiniBatch(ds[i:min(i+M,N)], M, apply_random_augmentations, K) for
#                   i in range(0,N,M)]
def batch(ds, N, M, K, augmentor = lambda x, k: x, shuffle=True):
    batched_ds = list()
    for i in range(0, N, M):
        j = min(i+M, N)
        minibatch = MiniBatch(ds[i:j], M, K, augmentor)
        batched_ds.append(minibatch)
    if shuffle:
        np.random.shuffle(batched_ds)
    return batched_ds

def HorizontalFlip(im, p=0.5):
    if p >= rng.random():
        return im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
    else:
        return im

def CropResize(im, crop_lo=0.08,crop_hi=1.0,aspect_lo=3/4,aspect_hi=4/3):
    new_aspect = rng.uniform(aspect_lo, aspect_hi)
    crop = rng.uniform(crop_lo, crop_hi)
    w,h = im.size
    s_w, s_h = (min(w, new_aspect * crop * w), crop * h)
    x1 = (w - s_w) * rng.random()
    x2 = x1 + s_w
    y1 = (h - s_h) * rng.random()
    y2 = y1 + s_h
    return im.resize((w,h),resample=2,box=(x1,y1,x2,y2))

def ColorJitter(im, b_max=0.8,c_max=0.8,s_max=0.8):
    rand = rng.random(3)
    im = PIL.ImageEnhance.Brightness(im).enhance(1 + rand[0] * b_max)
    im = PIL.ImageEnhance.Contrast(im).enhance(1 + rand[1] * c_max)
    im = PIL.ImageEnhance.Color(im).enhance(1 + rand[2] * s_max)
    return im

def Decolorize(im, p=0.2):
    if p >= rng.random():
        return PIL.ImageEnhance.Color(im).enhance(0.0)
    else:
        return im

def apply_random_augmentations(x, i,
    augmentations=[HorizontalFlip, ColorJitter, CropResize, Decolorize], channels=1):
    if channels == 1:
        mode = 'L'
    elif channels == 3:
        mode = 'RGB'
    w,h = x.shape
    im = PIL.Image.fromarray(x, mode)
    for A in augmentations:
        im = A(im)
    aug_x = np.array(im,dtype=np.float32).reshape(1,w,h,1)
    return aug_x

def aggregate_pos(X,M,K):
    aggX = list()
    aggT = list()
    # M(K^2-K) positive tuples
    for m in range(M):
        for k in range(K):
            x1 = X[m*K + k]
            for j in range(K):
                if j == k:  continue
                x2 = X[m*K + j]
                x = tf.concat((x1,x2), 0)
                aggX.append(x)
                aggT.append(1)
    return tf.stack(aggX), tf.stack(aggT)

def aggregate_neg(X, M, K):
    aggX = list()
    aggT = list()
    # M(K^2-K) negative tuples
    for m in range(M):
        for k in range(K):
            x1 = X[m*K + k]
            for i in range(K-1):
                neg_m = (m+i) % M
                neg_k = int(rng.random())*K
                x2 = X[neg_m*K + neg_k]
                x = tf.concat((x1,x2), 0)
                aggX.append(x)
                aggT.append(0)
    return tf.stack(aggX), tf.stack(aggT)

def aggregate(X, M, K):
    agg_pos = aggregate_pos(X,M,K)
    agg_neg = aggregate_neg(X,M,K)
    return tf.stack((agg_pos[0],agg_neg[0])), tf.stack((agg_pos[1],agg_neg[1]))

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
        return self.d2(x)

@tf.function
def loss(T, Y):
    return loss_object(T,Y)

def train_step(X, M, K, step):
    with tf.GradientTape(persistent=True) as tape:
        # encoding
        Z1 = backbone(tf.concat(X,0), training=True)
        # aggregation
        Z2, T = aggregate(Z1, M, K)
        # score tuples
        Y = relation(Z2, training=True)
        # approximate crossentropy
        l = loss(T, Y)
    if step == 0:
        relation_gradients = tape.gradient(l, relation.trainable_variables)
        optimizer.apply_gradients(zip(relation_gradients, relation.trainable_variables))
    elif step == 1:
        backbone_gradients = tape.gradient(l, backbone.trainable_variables)
        optimizer.apply_gradients(zip(backbone_gradients, backbone.trainable_variables))
    train_loss(T, Y)
    train_accuracy(T, Y)

def test_step(X, M, K):
    Z1 = backbone(tf.concat(X,0), training=False)
    Z2, T = aggregate(Z1, M, K)
    Y = relation(Z2, training=False)
    l = loss(T, Y)
    test_loss(T, Y)
    test_accuracy(T, Y)


def main():
    (train_x, _), (test_x, __) = tf.keras.datasets.fashion_mnist.load_data()

    #train_x = tf.random.shuffle(train_x)
    #test_x = tf.random.shuffle(test_x)

    backbone = Conv4()
    backbone.build((None,28,28,1))
    backbone.summary()
    relation = Relator()
    relation.build((None,128))
    relation.summary()

    loss_object = tf.keras.losses.BinaryCrossentropy()

    train_loss = tf.keras.metrics.BinaryCrossentropy()
    train_accuracy = tf.keras.metrics.BinaryAccuracy()
    test_loss = tf.keras.metrics.BinaryCrossentropy()
    test_accuracy = tf.keras.metrics.BinaryAccuracy()

    optimizer = tf.keras.optimizers.Adam(0.001)

    EPOCHS = 2
    N1 = len(train_x)
    N2 = len(test_x)
    M = 5
    K = 5
    p = 50
    for epoch in range(EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        batched_train_x = batch(train_x, N1, M, K, augmentor=apply_random_augmentations)
        batched_test_x = batch(test_x, N2, M, K, augmentor=apply_random_augmentations)

        print('training progress')
        for i in range(len(batched_train_x)):
            minibatch = batched_train_x[i]
            train_step(minibatch(), M, K, i%2)
            if i % (len(batched_train_x)//p) == 0:
                print(i/len(batched_train_x))

        print('testing...')
        for i in range(len(batched_test_x)):
            minibatch = batched_train_x[i]
            test_step(minibatch(), M, K)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
            )

    # TODO compare training dynamics w/ and w/out picky pairing (picky pairing checks for false positive and false negative tuples)

if __name__ == '__main__':
    main()
else:
    print('model.py')
