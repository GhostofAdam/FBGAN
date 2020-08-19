import tensorflow as tf
from load_data import charmap
import tflib.ops.conv1d as cv
import tflib.ops.linear as li

SEQ_LEN = 450
DIM = 21


def softmax(logits):
    return tf.reshape(
        tf.nn.softmax(tf.reshape(logits, [-1, len(charmap)])),
        tf.shape(logits)
    )


def ResBlock(name, inputs):
    output = inputs
    output = tf.nn.relu(output)
    output = cv.Conv1D(name+'.1', DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    output = cv.Conv1D(name+'.2', DIM, DIM, 5, output)
    return inputs + (0.3*output)


def Generator(z, y, isSupervised = True):
    output = tf.concat([z, y], 1)
    output = li.Linear('Generator.Input', 64 + 10, SEQ_LEN*DIM, output)
    output = tf.reshape(output, [-1, DIM, SEQ_LEN])
    output = ResBlock('Generator.1', output)
    output = ResBlock('Generator.2', output)
    output = ResBlock('Generator.3', output)
    output = ResBlock('Generator.4', output)
    output = ResBlock('Generator.5', output)
    output = cv.Conv1D('Generator.Output', DIM, len(charmap), 1, output)
    output = tf.transpose(output, [0, 2, 1])
    output = softmax(output)
    # print(output.shape)
    return output


def Discriminator(inputs, y, isSupervised = True):
    output = tf.transpose(inputs, [0, 2, 1])
    output = cv.Conv1D('Discriminator.Input', len(charmap), DIM, 1, output)
    output = ResBlock('Discriminator.1', output)
    output = ResBlock('Discriminator.2', output)
    output = ResBlock('Discriminator.3', output)
    output = ResBlock('Discriminator.4', output)
    output = ResBlock('Discriminator.5', output)
    output = tf.reshape(output, [-1, SEQ_LEN*DIM])
    output = tf.concat([output, y], 1)
    output = li.Linear('Discriminator.Output', SEQ_LEN*DIM + 10, 1, output)
    return output







