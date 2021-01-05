import tensorflow as tf
from focal_loss import SparseCategoricalFocalLoss

def get_optimizer(name, lr):
    """Returns an optimizer"""
    if name == 'momentum':
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)
    elif name == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif name == 'rmsprop':
        return tf.keras.optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError('Unknown optimizer {}'.format(name))

def get_loss(name):
    """Returns a loss function"""
    if name == 'crossentropy':
        return tf.keras.losses.SparseCategoricalCrossentropy()
    elif name == 'focalloss':
        return SparseCategoricalFocalLoss(gamma=2)
    else:
        raise ValueError('Unknown loss fn {}'.format(name))
