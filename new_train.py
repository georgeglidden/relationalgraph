import sys
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers

# hyperparameters
cifar_shape = (None,32,32,3)
cifar_shape_x = (1,32,32,3)
nfeatures = 64
nclasses = 10

rate = 1e-3
b1 = 0.9
b2 = 0.999
epochs = 100
M = batchsize = 64
K = 16
interval = 50
print(1)


# encoder
class Encoder(Model):
    def __init__(self, insh=cifar_shape, nb_features=nfeatures):
        super(Encoder, self).__init__()
        self.conv_0 = layers.Conv2D(64, kernel_size=3, strides=1,
                                     padding='same', activation='relu', input_shape=insh)
        self.batch_norm_0 = layers.BatchNormalization()
        self.conv_1 = layers.Conv2D(32, kernel_size=3, strides=2,
                                     padding='same', activation='relu')
        self.conv_2 = layers.Conv2D(32, kernel_size=3, strides=1,
                                     padding='same', activation='relu')
        self.batch_norm_2 = layers.BatchNormalization()
        self.conv_3 = layers.Conv2D(16, kernel_size=3, strides=2,
                                     padding='same', activation='relu')
        self.batch_norm_3 = layers.BatchNormalization()
        self.flatten = layers.Flatten()
        self.dense_features = layers.Dense(nb_features)

    def call(self, xin):
        x = self.conv_0(xin)
        x = self.batch_norm_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = self.flatten(x)
        return self.dense_features(x)

# relation head
class RelationHead(Model):
    def __init__(self, insh=(None,2*nfeatures)):
        super(RelationHead, self).__init__()
        self.dense_0 = layers.Dense(128, activation='relu', input_shape=insh)
        self.dense_1 = layers.Dense(64, activation='relu')
        self.batch_norm_1 = layers.BatchNormalization()
        self.dense_2 = layers.Dense(64, activation='relu')
        self.dense_3 = layers.Dense(32, activation='relu')
        self.dense_4 = layers.Dense(8, activation='relu')
        self.batch_norm_4 = layers.BatchNormalization()
        self.relation = layers.Dense(1, activation='sigmoid')

    def call(self, xin):
        x = self.dense_0(xin)
        x = self.dense_1(x)
        x = self.batch_norm_1(x)
        x = self.dense_2(x)
        x = self.dense_3(x)
        x = self.dense_4(x)
        x = self.batch_norm_4(x)
        return self.relation(x)

class SmallRel(Model):
    def __init__(self, insh=(None,2*nfeatures)):
        super(SmallRel, self).__init__()
        self.dense_in = layers.Dense(128, activation='relu', input_shape=insh)
        self.dense_1 = layers.Dense(256, activation='relu')
        self.batch_norm_1 = layers.BatchNormalization()
        self.dense_out = layers.Dense(1, activation='sigmoid')

    def call(self, xin):
        x = self.dense_in(xin)
        x = self.dense_1(x)
        x = self.batch_norm_1(x)
        return self.dense_out(x)

def randint(max=1, min=0):
    return int(min + (tf.random.uniform([1]) * max))

# batch augmentation and pair sampling
from numpy import min as np_min, max as np_max

def channel_norm(im):
    rows,cols,channels = im.shape
    for c in range(channels):
        channel_max = np_max(im[:,:,c])
        channel_min = np_min(im[:,:,c])
        if channel_max > channel_min: # prevents division by zero
            im[:,:,c] = (im[:,:,c] - channel_min) / (channel_max - channel_min)
    return im

# convert single channel images to rgb, return channelwise normalized array
def preprocess(im):
    if len(im.shape) == 2:
        im = tf.image.grayscale_to_rgb(im)
    return channel_norm(im)

# jitter hue,saturation,value,contrast
def color_jitter(im, h=0.5, s=0.3, v=0.2, c=0.2):
    im = tf.image.random_hue(im,h)
    im = tf.image.random_saturation(im, 1-s, 1+s)
    im = tf.image.random_brightness(im, v)
    return tf.image.random_contrast(im,1-c,1+c)

# convert a proportion p of inputs to 3-channel grayscale
def decolorize(im, p=0.5):
    if tf.random.uniform([1]) < p:
        return tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(im))
    else:
        return im

# crop and resize by bounded random parameters
def crop_resize(im, cl=0.3,ch=1.0,al=3/4,ah=4/3):
    h,w,channels = im.shape
    aspect,crop = tf.random.uniform([2])
    aspect = al + (ah-al)*aspect
    target_h = h * aspect
    target_w = w * (1/aspect)
    crop = cl + (ch-cl)*crop
    crop_h, crop_w = target_h * crop, target_w * crop
    crop_h, crop_w = int(min([crop_h, h-1])), int(min([crop_w, w-1]))
    im = tf.image.random_crop(im, [crop_h,crop_w,channels])
    return tf.image.resize(im, [h, w])

# apply a horizontal flip to some proportion p of inputs
def horizontal_flip(im, p=0.5):
    if tf.random.uniform([1]) < p:
        return tf.image.flip_left_right(im)
    else:
        return im

# apply some intensity of noise between lo and hi to the input image
def apply_noise(im, lo=0.0,hi=0.4):
    pass

def augmentation_sequence(im, numpy=False):
    if numpy == False:
        im = im.numpy()
    im = preprocess(im)
    im = color_jitter(im)
    im = decolorize(im)
    im = crop_resize(im)
    im = horizontal_flip(im)
    #im = apply_noise(im)
    return im

# apply all augment functions nb_augment times to each image in a batch
def augment(batch, nb_augment = 16):
    augmented_batch = []
    for i in range(batch.shape[0]):
        x = batch[i]
        augmented_batch.append(x)
        for j in range(1,nb_augment):
            aug_x = augmentation_sequence(x)
            augmented_batch.append(aug_x)
    return tf.stack(augmented_batch, 0)

from itertools import chain
# pair two input images
def pair_x(x1,x2,insh):#insh = cifar_shape_x
    if x1.shape != insh:
        x1 = tf.reshape(x1,insh)
    if x2.shape != insh:
        x2 = tf.reshape(x2, insh)
    return tf.concat([x1,x2], 0)

# aggregates two feature vectors together with a linear concatenation
def aggregate(z1,z2):
    return tf.concat([z1,z2], 0)
    #return tf.reshape(ag, (1,2*nfeatures))

# from M*K encoded augmented inputs, sample M*K(K-1)/2 positive and negative pairs
def positive_subsamples(encoded_batch, M, K):
    N = M * K
    return [encoded_batch[i:i+K] for i in range(0,N,K)]

def negative_subsamples(encoded_batch, M, K):
    N = M * K
    return [[encoded_batch[(i+j+j*K) % N] for j in range(K)]
                 for i in range(0,N,K)]

def sample_aggregate_pairs(subsamples, K):
    return list(chain(*[
        [aggregate(sub[i],sub[j]) for i in range(K) for j in range(i+1,K)] for sub in subsamples
    ]))

def sample(encoded_batch, M, K):
    pos = sample_aggregate_pairs(positive_subsamples(encoded_batch,M,K),K)
    neg = sample_aggregate_pairs(negative_subsamples(encoded_batch,M,K),K)
    return pos + neg

# dataset preparation
from tensorflow.keras.datasets import cifar10
(train_ds, _),(test_ds, __) = cifar10.load_data()
train_shape = train_ds.shape
test_shape = test_ds.shape

from tensorflow.data import Dataset
train_ds = Dataset.from_tensor_slices(train_ds)
train_ds = train_ds.shuffle(1000,reshuffle_each_iteration=True)
train_ds = train_ds.map(lambda x: x/255)
train_ds = train_ds.batch(batchsize)
test_ds = Dataset.from_tensor_slices(test_ds)
test_ds = test_ds.shuffle(1000,reshuffle_each_iteration=True)
test_ds = test_ds.map(lambda x: x/255)
test_ds = test_ds.batch(batchsize)
print(train_ds.cardinality())
print(test_ds.cardinality())

# train it

# refresh datasets
for _ in train_ds:
    pass
for _ in test_ds:
    pass
print(2)

# build models
encoder = Encoder()
encoder.build(cifar_shape)
relhead = SmallRel()
relhead.build((None,2*nfeatures))
print(3)

# do training

loss = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.Adam(rate, beta_1=b1, beta_2=b2)
metric_acc = tf.keras.metrics.BinaryAccuracy()
metric_loss = tf.keras.metrics.BinaryCrossentropy()

print(4)
for epoch in range(epochs):
    metric_acc.reset_states()
    metric_loss.reset_states()
    i = 0
    for batch in train_ds:
        i += 1
        augmented_batch = augment(batch, K)
        with tf.GradientTape() as tape:
            encoded_batch = encoder(augmented_batch, training=True)
            pairs = sample(encoded_batch, M, K)
            pairs_tensor = tf.stack(pairs)
            scores = relhead(pairs_tensor, training=True)
            targets = [1] * (len(pairs)//2) + [0] * (len(pairs)//2)
            err = loss(targets,scores)
        encoder_grad, relhead_grad = tape.gradient(err,
            [encoder.trainable_weights,relhead.trainable_weights])
        optimizer.apply_gradients(zip(encoder_grad, encoder.trainable_weights))
        optimizer.apply_gradients(zip(relhead_grad, relhead.trainable_weights))
        metric_acc.update_state(targets,scores)
        metric_loss.update_state(targets,scores)
        if i % interval == 0:
            print(
                f'{i} '
                f'loss: {metric_loss.result()}'
                f' acc: {metric_acc.result()}'
            )
