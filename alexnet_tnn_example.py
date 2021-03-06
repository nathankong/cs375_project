import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tnn import main
from tnnrnn import tnn_ConvLSTMCell, tnn_DenseRNNCell
import numpy as np

batch_size = 256 # batch size for training
NUM_CATS = 1000 # number of categories
NUM_TIMESTEPS = 3 # number of timesteps we are predicting on
NETWORK_DEPTH = 8 # number of total layers in our network
TOTAL_TIMESTEPS = NETWORK_DEPTH + NUM_TIMESTEPS # we always unroll num_timesteps after the first output of the model
TRAIN = True # the train flag, SET THIS to False if evaluating the loss

# we unroll at least NETWORK_DEPTH times (8 in this case) so that the input can reach the output of the network
# note tau is the value of the memory decay (by default 0) at the fc layers (due to GPU memory constraints) and trainable_flag is whether the memory decay is trainable, which by default is False

def model_func(input_images, train=True, ntimes=TOTAL_TIMESTEPS, batch_size=batch_size, edges_arr=[], base_name='alexnet_tnn', tau=0.0, trainable_flag=False, channel_op='concat'):
    with tf.variable_scope("alexnet_tnn"):
        base_name += '.json'
        print('Using base: ', base_name)
        # creates the feedforward network graph from json
        G = main.graph_from_json(base_name)

        for node, attr in G.nodes(data=True):
            if node in ['fc6', 'fc7']:
                if train: # we add dropout to fc6 and fc7 during training
                    print('Using dropout for ' + node)
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 0.5
                else: # turn off dropout during training
                    print('Not using dropout')
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 1.0

            memory_func, memory_param = attr['kwargs']['memory']
            if 'filter_size' in memory_param:
                attr['cell'] = tnn_ConvLSTMCell
            else:
                attr['kwargs']['memory'][1]['memory_decay'] = tau
                attr['kwargs']['memory'][1]['trainable'] = trainable_flag

        # add any non feedforward connections here: [('L2', 'L1')]
        G.add_edges_from(edges_arr)

        # initialize network to infer the shapes of all the parameters
        main.init_nodes(G, input_nodes=['conv1'], batch_size=batch_size, channel_op=channel_op)
        # unroll the network through time
        main.unroll(G, input_seq={'conv1': input_images}, ntimes=ntimes)

        outputs = {}
        # start from the final output of the model and num timesteps beyond that
        # for t in range(ntimes-NUM_TIMESTEPS, ntimes):
        #     idx = t - (ntimes - NUM_TIMESTEPS) # keys start at timepoint 0
        #    outputs[idx] = G.node['fc8']['outputs'][t]
        
        # alternatively, we return the final output of the model at the last timestep
        outputs[0] = G.node['fc8']['outputs'][-1]
        return outputs

# get random images of size 224, 224, 3 (this is where the imagenet images would be)
# typically they are 256, 256, 3, but we resize them using tf.resize_images to 224

# create the model
x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])

y_ = tf.placeholder(tf.int32, [batch_size,]) # predicting a single label for each input image

logits = model_func(x, train=TRAIN, ntimes=TOTAL_TIMESTEPS, batch_size=batch_size, edges_arr=[], base_name='alexnet_tnn', tau=0.0, trainable_flag=False, channel_op='concat')

# setup the loss (average across time, the cross entropy loss at each timepoint between model predictions and ground truth categories)
with tf.name_scope('cumulative_loss'):
    cumm_loss = tf.reduce_mean(tf.add_n([tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y_) for logit in logits.values()]) / len(logits))

# setup the optimizer
with tf.name_scope('adam_optimizer'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cumm_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch_img = np.random.randn(batch_size, 224, 224, 3)
        labels_batch = np.random.randint(1000, size=batch_size) # random ints between 0-999 for the labels as they are categories
        if i % 100 == 0:
            TRAIN = False # set this to false for evaluation
            train_loss = cumm_loss.eval(feed_dict={x: batch_img, y_: labels_batch})
            print('step %d, training loss %g' % (i, train_loss))
            TRAIN = True # set this back to true for training
        train_step.run(feed_dict={x: batch_img, y_: labels_batch})

