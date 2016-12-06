# Convolutional neural network for paper texture classification
# A.G. Klein v0.0 22-Oct-2016 -- initial version

# Note: While 256x256 and 128x128 images were tested, 64x64 images
# gave similar performance, so I use those for simplicity.  Each
# image file is chopped into 25 tiles (4x4, then another 3x3 taken
# by removing 32 pixels around the edges of each image to induce 
# an offset, and subsequently horizontal flipping).  The validation
# set is created by randomly sampling the training set, being
# careful not to comingle source images across sets.  Code to load
# training/validation sets and labels can be found in lines 75-83.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy
from six.moves import xrange  # pylint: disable=redefined-builtin
import scipy.io as sio
import tensorflow as tf

INPUT_FILE = 'dataset_full64_MoMA.mat'  # Matlab file where results get loaded
IMAGE_SIZE = 64
NUM_LABELS = 7
TILES_PER_IMAGE=25;          # each image file was presumed split into this many distinct tiles
SEED = 16455                 # set to None for random seed.
BATCH_SIZE = 32              # batch size
NUM_EPOCHS = 43.015          # number of times we go through all the data
EVAL_BATCH_SIZE = 100        # size of each evaluation batch (might affect speed slightly)
EVAL_FREQUENCY = 100         # number of steps between evaluations (how often we print results)
PATCH_SIZE = 5               # size of patches in CNN layers (they'll be PATCH_SIZE x PATCH_SIZE)
CL1_SIZE = 16                # number of features in 1st CNN layer
CL2_SIZE = 32                # number of features in 2nd CNN layer
CL3_SIZE = 64                # number of features in 3rd CNN layer
CL4_SIZE = 128               # number of features in 4th CNN layer
CL1_maxpool = 8              # downsampling factor after 1st CNN
CL2_maxpool = 2              # downsampling factor after 2nd CNN
CL3_maxpool = 2              # downsampling factor after 3rd CNN
CL4_maxpool = 2              # downsampling factor after 4th CNN
FC_SIZE = 1024               # size of fully-connected layer
BASE_LEARNING_RATE = 0.001   # starting learning rate
LEARNING_DECAY_RATE = 0.95   # exponential learning decay rate

def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - ( 100.0 * numpy.sum(numpy.argmax(predictions, 1) == labels) / predictions.shape[0])

def error_rate_voting(predictions, labels):
  # currently, each large image consists of multiple tiles with a label for each. 
  # reduce to a single label per larger image.
  labels=labels[::TILES_PER_IMAGE]
  
  # do 'voting' over all tiles for each image
  predictions2 = numpy.ndarray(shape=(labels.shape[0], NUM_LABELS), dtype=numpy.float32)
  for i in xrange(0, labels.shape[0]):
    predictions2[i,:]=numpy.mean(predictions[i*TILES_PER_IMAGE:(i+1)*TILES_PER_IMAGE,:],0)
  predictions2=numpy.argmax(predictions2, 1)
  
  # since we had split original group 1 into two different groups 
  # called group 0 and group 1, recombine them (i.e., set all
  # group 0 --> group 1)
  predictions2[predictions2==0]=1
  labels[labels==0]=1;
  
  # compute error
  return 100.0 - ( 100.0 * numpy.sum(predictions2 == labels) / predictions2.shape[0])

def main(argv=None):  # pylint: disable=unused-argument

  # Extract train/test data from Matlab file into numpy arrays.
  mat_contents = sio.loadmat(INPUT_FILE)
  train_data=numpy.expand_dims(mat_contents['train_data'], axis=3)
  train_labels=mat_contents['train_labels'].squeeze()
  train_fnames=mat_contents['train_fnames'].squeeze()
  train_size = train_labels.shape[0]
  validation_data=numpy.expand_dims(mat_contents['validation_data'], axis=3)  # randomly selected images from training set
  validation_labels=mat_contents['validation_labels'].squeeze()
  validation_fnames=mat_contents['validation_fnames'].squeeze()
  validation_size = validation_labels.shape[0]

  # Feed training samples and labels to the graph.
  train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))
  train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
  eval_data = tf.placeholder(tf.float32, shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1))

  # Initialize trainable weights
  conv1_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, 1, CL1_SIZE], stddev=0.1, seed=SEED, dtype=tf.float32))
  conv1_biases = tf.Variable(tf.zeros([CL1_SIZE], dtype=tf.float32))
  conv2_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, CL1_SIZE, CL2_SIZE], stddev=0.1, seed=SEED, dtype=tf.float32))
  conv2_biases = tf.Variable(tf.constant(0.1, shape=[CL2_SIZE], dtype=tf.float32))
  conv3_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, CL2_SIZE, CL3_SIZE], stddev=0.1, seed=SEED, dtype=tf.float32))
  conv3_biases = tf.Variable(tf.constant(0.1, shape=[CL3_SIZE], dtype=tf.float32))
  conv4_weights = tf.Variable(tf.truncated_normal([PATCH_SIZE, PATCH_SIZE, CL3_SIZE, CL4_SIZE], stddev=0.1, seed=SEED, dtype=tf.float32))
  conv4_biases = tf.Variable(tf.constant(0.1, shape=[CL4_SIZE], dtype=tf.float32))
  downsampling_factor = CL1_maxpool * CL2_maxpool * CL3_maxpool * CL4_maxpool;
  fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // downsampling_factor * IMAGE_SIZE // downsampling_factor * CL4_SIZE, FC_SIZE], stddev=0.1, seed=SEED, dtype=tf.float32))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[FC_SIZE], dtype=tf.float32))
  fc2_weights = tf.Variable(tf.truncated_normal([FC_SIZE, NUM_LABELS], stddev=0.1, seed=SEED, dtype=tf.float32))
  fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=tf.float32))

  # Define model
  def model(data, train=False):

    # CNN Layer 1
    conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME'), conv1_biases))
    pool = tf.nn.max_pool(conv, ksize=[1, CL1_maxpool, CL1_maxpool, 1], strides=[1, CL1_maxpool, CL1_maxpool, 1], padding='SAME')

    # CNN Layer 2
    conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool, conv2_weights, strides=[1, 1, 1, 1], padding='SAME'), conv2_biases))
    pool = tf.nn.max_pool(conv, ksize=[1, CL2_maxpool, CL2_maxpool, 1], strides=[1, CL2_maxpool, CL2_maxpool, 1], padding='SAME')

    # CNN Layer 3
    conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool, conv3_weights, strides=[1, 1, 1, 1], padding='SAME'), conv3_biases))
    pool = tf.nn.max_pool(conv, ksize=[1, CL3_maxpool, CL3_maxpool, 1], strides=[1, CL3_maxpool, CL3_maxpool, 1], padding='SAME')

    # CNN Layer 4
    conv = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool, conv4_weights, strides=[1, 1, 1, 1], padding='SAME'), conv4_biases))
    pool = tf.nn.max_pool(conv, ksize=[1, CL4_maxpool, CL4_maxpool, 1], strides=[1, CL4_maxpool, CL4_maxpool, 1], padding='SAME')

    # Reshape the feature map cuboid into matrix, do fully connected layers
    pool_shape = pool.get_shape().as_list()
    hidden = tf.nn.relu(tf.matmul(tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]]), fc1_weights) + fc1_biases)
    # 50% dropout during training
    if train:
      hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
    return tf.matmul(hidden, fc2_weights) + fc2_biases

  # Training computation: logits + cross-entropy loss.
  logits = model(train_data_node, True)
  loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels_node))

  # L2 regularization for the fully connected parameters.
  regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) + tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))

  # Add regularization term to the loss.
  loss += 5e-4 * regularizers

  # Optimizer: set up a variable that's incremented once per batch and controls the learning rate decay.
  batch = tf.Variable(0, dtype=tf.float32)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      BASE_LEARNING_RATE,  # Base learning rate.
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,          # Decay step.
      LEARNING_DECAY_RATE, # Decay rate.
      staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(model(eval_data))

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  # Replace the following line with this one to force run on CPU (i.e., to get repeatable numbers used in paper)
  #with tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})) as sess:
  with tf.Session() as sess:
    tf.initialize_all_variables().run()

    # Do the training in "minibatches"
    for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data, train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.

      _, l, lr, predictions = sess.run([optimizer, loss, learning_rate, train_prediction], feed_dict=feed_dict)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' % (step, float(step) * BATCH_SIZE / train_size, 1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))        
        sys.stdout.flush()
    # Print the result
    print('\n** Results summary **')
    print('Validation error with voting: %.1f%%' % error_rate_voting(eval_in_batches(validation_data, sess), validation_labels))

if __name__ == '__main__':
  tf.app.run()

