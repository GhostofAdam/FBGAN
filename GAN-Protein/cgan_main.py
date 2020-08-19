import os
import sys
import time
import numpy as np
import tensorflow as tf
import language_helpers
import tflib as lib
import tflib.plot
from load_data import LoadData, data_real_gen
from load_data import charmap, inv_charmap
from cgan_model import Generator, Discriminator
sys.path.append(os.getcwd())

# Hyperparameters
BATCH_SIZE = 50                                # Batch size
ITERS = 50000                                  # How many iterations to train for
SEQ_LEN = 450                                  # Sequence length in characters
DIM = 21                                       # Model dimensionality
CRITIC_ITERS = 5                               # How many critic iterations per generator iteration
LAMBDA = 10                                    # Gradient penalty lambda hyperparameter.
# lib.print_model_settings(locals().copy())


# Construct Model
def model(lines_ch, lines):
    # Construct Input data
    real_inputs = tf.placeholder(tf.float32, shape=[BATCH_SIZE, SEQ_LEN, len(charmap)])
    real_label = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 10])

    fake_inputs = Generator(tf.random_normal(shape=[n_samples, 64]), real_label)

    # Input Discriminator
    disc_real = Discriminator(real_inputs, real_label)
    disc_fake = Discriminator(fake_inputs, real_label)

    # Compute D/G cost
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
    gen_cost = -tf.reduce_mean(disc_fake)

    # WGAN-GP L constraints
    alpha = tf.random_uniform(shape=[BATCH_SIZE, 1, 1], minval=0., maxval=1.)
    differences = fake_inputs - real_inputs
    interpolates = real_inputs + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1, 2]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    # Get parameters
    gen_params = lib.params_with_name('Generator')
    disc_params = lib.params_with_name('Discriminator')

    # Construct optimizer
    # Optimizer？
    gen_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=disc_params)

    # During training we monitor JS divergence between the true & generated ngram distributions for n=1,2,3,4.// 2/3/4/5
    # To get an idea of the optimal values, we evaluate these statistics on a held-out set first.
    true_char_ngram_lms = [language_helpers.NgramLanguageModel(i, lines_ch[100*BATCH_SIZE:], tokenize=False) for i in range(2, 6)]
    validation_char_ngram_lms = [language_helpers.NgramLanguageModel(i, lines_ch[:100*BATCH_SIZE], tokenize=False) for i in range(2, 6)]
    for i in range(0, 4):
        print("validation set JSD for n={}: {}".format(i+2, true_char_ngram_lms[i].js_with(validation_char_ngram_lms[i])))
    true_char_ngram_lms = [language_helpers.NgramLanguageModel(i, lines_ch[:], tokenize=False) for i in range(2, 6)]

    # Start run the graph
    saver = tf.train.Saver()
    with tf.Session() as session:
        # initialize
        session.run(tf.initialize_all_variables())

        # Generate fake samples
        def generate_samples():
            samples = session.run(fake_inputs)
            samples = np.argmax(samples, axis=2)
            # print(samples.shape)
            decoded_samples = []
            for i in range(len(samples)):
                decoded = ''
                for j in samples[i]:
                    decoded += inv_charmap[j]
                decoded_samples.append(decoded)
            # print(len(decoded_samples), len(decoded_samples[0]), decoded_samples[0])
            return decoded_samples

        # Generate real samples
        gen = data_real_gen(lines)

        # Start iteration
        for iteration in range(ITERS):
            start_time = time.time()
            # Train generator
            _gen_cost = 0
            if iteration > 0:
                _gen_cost, _ = session.run([gen_cost, gen_train_op])
            # Train discriminator
            _disc_cost = 0
            for i in range(CRITIC_ITERS):
                _data = next(gen)
                _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={real_inputs: _data})
            # Plot curve
            lib.plot.plot('time', time.time() - start_time)
            lib.plot.plot('train gen cost', _gen_cost)
            lib.plot.plot('train disc cost', _disc_cost)

            if iteration % 100 == 99:
                saver.save(session, 'checkpoint/Mymodel_'+str(iteration))
                samples = []
                for i in range(20):
                    samples.extend(generate_samples())
                for i in range(0, 4):
                    lm = language_helpers.NgramLanguageModel(i+2, samples, tokenize=False)
                    lib.plot.plot('js{}'.format(i+2), true_char_ngram_lms[i].js_with(lm))
                with open('samples_{}.txt'.format(iteration), 'w', encoding="utf-8") as f:
                    for s in samples:
                        s = "".join(s)
                        f.write(s + "\n")
            if iteration % 100 == 99:
                lib.plot.flush()
            lib.plot.tick()


if __name__ == '__main__':

    # configuration GPU enviroment
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    session = tf.Session(config=config)
    # Load data
    lines_ch, lines = LoadData()
    # Run model
    model(lines_ch, lines)

