from model import Conv4, ConvLong, Relator, batch, train_step, test_step, aggregate
from generate_pairs import pair
from augmentations import apply_all, no_aug
import tensorflow as tf
from tensorflow.keras import datasets as keras_datasets
import sys
from time import time
from numpy import mean

def load_ds(ds_name):
    print('loading', ds_name)
    if ds_name == 'F-MNIST':
        return keras_datasets.fashion_mnist.load_data()
    elif ds_name == 'MNIST':
        return keras_datasets.mnist.load_data()
    elif ds_name == 'CIFAR10':
        return keras_datasets.cifar10.load_data()
    elif ds_name == 'CIFAR100':
        return keras_datasets.cifar100.load_data()

def prep_ds(dataset):
    (train_x, train_y), (test_x, test_y) = dataset
    train_perm = tf.random.shuffle(tf.range(tf.shape(train_x)[0]))
    train_x = tf.gather(train_x, train_perm, axis=0)
    train_y = tf.gather(train_y, train_perm, axis=0)
    test_perm = tf.random.shuffle(tf.range(tf.shape(test_x)[0]))
    test_x = tf.gather(test_x, test_perm, axis=0)
    test_y = tf.gather(test_y, test_perm, axis=0)
    return (train_x, train_y), (test_x, test_y)

def build_model(name, dataset, size, rate):
    if name =='Conv4':
        backend = Conv4
    elif name == 'ConvLong':
        backend = ConvLong
    part_a = backend(size)
    if dataset == 'CIFAR10' or dataset == 'CIFAR100':
        ins = (None,32,32,3)
    elif dataset == 'MNIST' or dataset == 'F-MNIST':
        ins = (None,28,28,1)
    part_a.build(ins)
    part_b = Relator()
    part_b.build((None,128))
    loss = tf.keras.losses.BinaryCrossentropy()
    optimizer = tf.keras.optimizers.Adam(rate)
    return [part_a, part_b], loss, optimizer

def setup_augbatch(do_aug, mbatch_size, nb_aug):
    if int(do_aug) == 1:
        augment_opt = apply_all
    else:
        augment_opt = no_aug
    M = int(mbatch_size)
    K = int(nb_aug)
    return augment_opt, M, K

def main():
    state = {
        'dataset':sys.argv[1],
        'model':sys.argv[2],
        'model size':sys.argv[3],
        'epochs':sys.argv[4],
        'training rate':sys.argv[5],
        'mbatch size':sys.argv[6],
        'do aug':sys.argv[7],
        'nb aug':sys.argv[8]
    }

    print('state\n', state)

    PORT = 465
    id = sys.argv[9]
    pr = int(sys.argv[10])

    # load and prep data
    dataset = load_ds(state['dataset'])
    (train_x, train_y), (test_x, test_y) = prep_ds(dataset)
    N1 = len(train_x)
    N2 = len(test_x)

    # build model
    (model_a, model_b), loss, optimizer = build_model(state['model'], state['dataset'], int(state['model size']), float(state['training rate']))
    summarylist = []
    model_a.summary(print_fn=lambda x: summarylist.append(x))
    model_a_summary = "\n".join(summarylist)
    summarylist = []
    model_b.summary(print_fn=lambda x: summarylist.append(x))
    model_b_summary = "\n".join(summarylist)

    # define augmentor, minibatch params
    augment_opt, M, K = setup_augbatch(state['do aug'], state['mbatch size'], state['nb aug'])

    # do training
    nb_train_steps = N1 // M
    p = M * (nb_train_steps // pr)
    s = M * (nb_train_steps // 20) # save every 5%
    nb_test_steps = N2 / M
    p2 = M * (nb_test_steps // 10)
    stats_list = []
    train_accuracy = 0.0
    train_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
    train_loss_metric = tf.keras.metrics.BinaryCrossentropy()
    test_accuracy = 0.0
    test_accuracy_metric = tf.keras.metrics.BinaryAccuracy()
    print('M, K:', (M,K))
    print('P, P2:', (p, p2))
    for epoch in range(int(state['epochs'])):
        telapsed_batch = []
        telapsed_eval = []
        telapsed_grad = []
        telapsed_end = []
        train_accuracy_metric.reset_states()
        train_loss_metric.reset_states()
        test_accuracy_metric.reset_states()
        train_batches = batch(train_x, train_y, N1, M, K, augmentor=augment_opt)
        test_batches = batch(test_x, test_y, N2, M, K, augmentor=augment_opt)

        print('training')
        i = 0
        for minibatch in train_batches:
            step_start = time()
            i += M
            train_batch_x, train_batch_y = minibatch()
            step_batch = time()
            with tf.GradientTape() as tape:
                Z1 = model_a(train_batch_x, training=True)
                Z2, T2 = pair(Z1, M, K)
                Y = model_b(Z2, training=True)
                loss_value = loss(T2, Y)
            step_evaluate = time()
            grads_a, grads_b = tape.gradient(loss_value, [model_a.trainable_weights, model_b.trainable_weights])
            optimizer.apply_gradients(zip(grads_a, model_a.trainable_weights))
            optimizer.apply_gradients(zip(grads_b, model_b.trainable_weights))
            step_gradopt = time()

            train_accuracy_metric.update_state(T2, Y)
            train_loss_metric.update_state(T2,Y)
            if i % s == 0:
                print('network savepoint')
                model_a.save(f'model_tests/id{id}_encoder_epoch{epoch}_step{i}_loss{str(train_loss_metric.result())[:5]}')
                model_b.save(f'model_tests/id{id}_relator_epoch{epoch}_step{i}_loss{str(train_loss_metric.result())[:5]}')
            if i % p == 0:
                print(
                f'{100*i/N1}% '
                f'loss {loss_value} '
                f'acc {train_accuracy_metric.result()} '
                f'avg loss {train_loss_metric.result()}\n'
                f'{tf.reshape(Y-T2, [-1])}')
            step_end = time()
            telapsed_batch.append(step_batch - step_start)
            telapsed_eval.append(step_evaluate - step_batch)
            telapsed_grad.append(step_gradopt - step_evaluate)
            telapsed_end.append(step_end - step_gradopt)
            print(f'time to unpack batch: {step_batch - step_start}\n'
                f'time to evaluate: {step_evaluate - step_batch}\n'
                f'time to calc grad and optimize: {step_gradopt - step_evaluate}\n'
                f'time from optimization to end: {step_end - step_gradopt}\n\n')

        print(f'avg unpack batch: {mean(telapsed_batch)}\n'
            f'avg evaluate: {mean(telapsed_eval)}\n'
            f'avg calc grad and optimize: {mean(telapsed_grad)}\n'
            f'avg end: {mean(telapsed_end)}\n\n')
        print('testing')
        i = 0
        for minibatch in test_batches:
            i += M
            test_batch_x, test_batch_y = minibatch()
            Z1 = model_a(test_batch_x, training=True)
            Z2, T2 = aggregate(Z1, test_batch_y, M, K)
            Y = model_b(Z2, training=True)
            test_accuracy_metric.update_state(T2,Y)
            if i % p2 == 0:
                print(
                f'{100*i/N2}%')
        print(f'test accuracy: {test_accuracy_metric.result()}')

if __name__ == '__main__':
    main()
