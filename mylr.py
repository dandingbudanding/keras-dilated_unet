from keras.callbacks import LearningRateScheduler
lr_base = 0.001
epochs = 250
lr_power = 0.9
def lr_scheduler(epoch, mode='power_decay'):
    '''if lr_dict.has_key(epoch):
        lr = lr_dict[epoch]
        print 'lr: %f' % lr'''

    if mode is 'power_decay':
        # original lr scheduler
        lr = lr_base * ((1 - float(epoch) / epochs) ** lr_power)
    if mode is 'exp_decay':
        # exponential decay
        lr = (float(lr_base) ** float(lr_power)) ** float(epoch + 1)
    # adam default lr
    if mode is 'adam':
        lr = 0.001

    if mode is 'progressive_drops':
        # drops as progression proceeds, good for sgd
        if epoch > 0.9 * epochs:
            lr = 0.0001
        elif epoch > 0.75 * epochs:
            lr = 0.001
        elif epoch > 0.5 * epochs:
            lr = 0.01
        else:
            lr = 0.1

    print('lr: %f' % lr)
    return lr

