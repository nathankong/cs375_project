from __future__ import division, print_function, absolute_import
import os, sys
from collections import OrderedDict
import numpy as np
import tensorflow as tf
from tfutils import base, data, model, optimizer, utils
import copy

from tnn import main
from tnnrnn import tnn_ConvLSTMCell, tnn_DenseRNNCell

## THINGS TO CHANGE!
# toggle this to train or to validate at the end
train_net = True
# toggle this to train on whitenoise or naturalscene data
#stim_type = 'whitenoise'
stim_type = 'naturalscene'

NUM_TIMESTEPS = 40 # number of timesteps we are predicting on
NETWORK_DEPTH = 3 # number of total layers in our network
TOTAL_TIMESTEPS = NETWORK_DEPTH + NUM_TIMESTEPS # we always unroll num_timesteps after the first output of the model
#TOTAL_TIMESTEPS = 40

# Figure out the hostname
host = os.uname()[1]
if 'instance-1' in host:
    if train_net:
        print('In train mode...')
        #TOTAL_BATCH_SIZE = 5000
        #MB_SIZE = 5000
        #TOTAL_BATCH_SIZE = 100
        #MB_SIZE = 100
        # Change batch size for ConvLstm1->ConvLstm2->Fc model
        TOTAL_BATCH_SIZE = 75
        MB_SIZE = 75
        NUM_GPUS = 1
    else:
        print('In val mode...')
        if stim_type == 'whitenoise':
            TOTAL_BATCH_SIZE = 5957
            MB_SIZE = 5957
            NUM_GPUS = 1
        else:
            #TOTAL_BATCH_SIZE = 5956
            #MB_SIZE = 5956
            #TOTAL_BATCH_SIZE = 100
            #MB_SIZE = 100
            # Change batch size for ConvLstm1->ConvLstm2->Fc model
            TOTAL_BATCH_SIZE = 75
            MB_SIZE = 75
            NUM_GPUS = 1
            
else:
    print("Data path not found!!")
    exit()

if not isinstance(NUM_GPUS, list):
    DEVICES = ['/gpu:' + str(i) for i in range(NUM_GPUS)]
else:
    DEVICES = ['/gpu:' + str(i) for i in range(len(NUM_GPUS))]

MODEL_PREFIX = 'model_0'

# Data parameters
if stim_type == 'whitenoise':
    N_TRAIN = 323762
    N_TEST = 5957
else:
    N_TRAIN = 323756
    N_TEST = 5956

#INPUT_BATCH_SIZE = 1024 # queue size
INPUT_BATCH_SIZE = 1000 # queue size
OUTPUT_BATCH_SIZE = TOTAL_BATCH_SIZE
print('TOTAL BATCH SIZE:', OUTPUT_BATCH_SIZE)
NUM_BATCHES_PER_EPOCH = N_TRAIN // OUTPUT_BATCH_SIZE
IMAGE_SIZE_RESIZE = 50

DATA_PATH = '/datasets/deepretina_data/tf_records/' + stim_type
print('Data path: ', DATA_PATH)

# data provider
class retinaTF(data.TFRecordsParallelByFileProvider):
    def __init__(self,
                 source_dirs,
                 resize=IMAGE_SIZE_RESIZE,
                 **kwargs):
        if resize is None:
            self.resize = 50
        else:
            self.resize = resize

        postprocess = {'images': [], 'labels': []}
        postprocess['images'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['images'].insert(1, (tf.reshape, ([-1] + [50, 50, 40], ), {}))
        postprocess['images'].insert(2, (self.postproc_imgs, (), {})) 
    
        postprocess['labels'].insert(0, (tf.decode_raw, (tf.float32, ), {})) 
        postprocess['labels'].insert(1, (tf.reshape, ([-1] + [5], ), {}))

        super(retinaTF, self).__init__(
              source_dirs,
              postprocess=postprocess,
              **kwargs)


    def postproc_imgs(self, ims):
        def _postprocess_images(im):
            im = tf.image.resize_images(im, [self.resize, self.resize])
            return im
        return tf.map_fn(lambda im: _postprocess_images(im), ims, dtype=tf.float32)

# Default convLSTM
def model_func(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, ntimes=TOTAL_TIMESTEPS, edges_arr=[], base_name='retina_tnn', tau=0.0, trainable_flag=False, channel_op='concat', seed=0, cfg_final=None):

    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]

    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size

    input_images = inputs['images']
    input_images = tf.reshape(input_images, [batch_size, 40, 50, 50, 1])

    input_list = []
    for i in range(40):
        slice_val = tf.squeeze(tf.slice(input_images, [0, i, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        input_list.append(slice_val)
    input_list.append(slice_val)
    input_list.append(slice_val)
    input_list.append(slice_val)

    with tf.variable_scope("retina_tnn"):
        base_name += '.json'
        print('Using base: ', base_name)
        # creates the feedforward network graph from json
        G = main.graph_from_json(base_name)

        for node, attr in G.nodes(data=True):
            memory_func, memory_param = attr['kwargs']['memory']
            if 'filter_size' in memory_param:
                attr['cell'] = tnn_ConvLSTMCell
            elif 'nunits' in memory_param:
                attr['cell'] = tnn_DenseRNNCell
            else:
                attr['kwargs']['memory'][1]['memory_decay'] = tau
                attr['kwargs']['memory'][1]['trainable'] = trainable_flag

        # add any non feedforward connections here: [('L2', 'L1')]
        G.add_edges_from(edges_arr)

        # initialize network to infer the shapes of all the parameters
        main.init_nodes(G, input_nodes=['conv1'], batch_size=batch_size, channel_op=channel_op)
        # unroll the network through time
        main.unroll(G, input_seq={'conv1': input_list}, ntimes=ntimes)

        outputs = {}
        # start from the final output of the model and num timesteps beyond that
        # for t in range(ntimes-NUM_TIMESTEPS, ntimes):
        #     idx = t - (ntimes - NUM_TIMESTEPS) # keys start at timepoint 0
        #    outputs[idx] = G.node['fc8']['outputs'][t]
        
        # alternatively, we return the final output of the model at the last timestep
        outputs['pred'] = G.node['fc3']['outputs'][-1]
        return outputs, params

# Default convLstm but with dropout (this is the one used that is trained)
def convLstmDropout(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, ntimes=TOTAL_TIMESTEPS, edges_arr=[], base_name='retina_tnn_dropout', tau=0.0, trainable_flag=False, channel_op='concat', seed=0, cfg_final=None):
    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]

    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size

    input_images = inputs['images']
    input_images = tf.reshape(input_images, [batch_size, 40, 50, 50, 1])

    input_list = []
    for i in range(40):
        slice_val = tf.squeeze(tf.slice(input_images, [0, i, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        input_list.append(slice_val)
    input_list.append(slice_val)
    input_list.append(slice_val)
    input_list.append(slice_val)

    with tf.variable_scope("retina_tnn_dropout"):
        base_name += '.json'
        print('Using base: ', base_name)
        # creates the feedforward network graph from json
        G = main.graph_from_json(base_name)

        for node, attr in G.nodes(data=True):
            if node in ['conv1', 'conv2']:
                if train: # we add dropout to fc6 and fc7 during training
                    print('Using dropout for ' + node)
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 0.75
                else: # turn off dropout during training
                    print('Not using dropout')
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 1.0

            memory_func, memory_param = attr['kwargs']['memory']
            if 'filter_size' in memory_param:
                attr['cell'] = tnn_ConvLSTMCell
            elif 'nunits' in memory_param:
                assert(0)
                attr['cell'] = tnn_DenseRNNCell
            else:
                attr['kwargs']['memory'][1]['memory_decay'] = tau
                attr['kwargs']['memory'][1]['trainable'] = trainable_flag

        # add any non feedforward connections here: [('L2', 'L1')]
        G.add_edges_from(edges_arr)

        # initialize network to infer the shapes of all the parameters
        main.init_nodes(G, input_nodes=['conv1'], batch_size=batch_size, channel_op=channel_op)
        # unroll the network through time
        main.unroll(G, input_seq={'conv1': input_list}, ntimes=ntimes)

        outputs = {}
        # start from the final output of the model and num timesteps beyond that
        # for t in range(ntimes-NUM_TIMESTEPS, ntimes):
        #     idx = t - (ntimes - NUM_TIMESTEPS) # keys start at timepoint 0
        #    outputs[idx] = G.node['fc8']['outputs'][t]
        
        # alternatively, we return the final output of the model at the last timestep
        outputs['pred'] = G.node['fc3']['outputs'][-1]
        return outputs, params

def convLstmDropout_no_pre_mem(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, ntimes=TOTAL_TIMESTEPS, edges_arr=[], base_name='retina_tnn_dropout_no_conv', tau=0.0, trainable_flag=False, channel_op='concat', seed=0, cfg_final=None):
    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]

    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size

    input_images = inputs['images']
    input_images = tf.reshape(input_images, [batch_size, 40, 50, 50, 1])

    input_list = []
    for i in range(40):
        slice_val = tf.squeeze(tf.slice(input_images, [0, i, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        input_list.append(slice_val)
    input_list.append(slice_val)
    input_list.append(slice_val)
    input_list.append(slice_val)

    with tf.variable_scope("retina_tnn_dropout_no_conv"):
        base_name += '.json'
        print('Using base: ', base_name)
        # creates the feedforward network graph from json
        G = main.graph_from_json(base_name)

        for node, attr in G.nodes(data=True):
            if node in ['conv1', 'conv2']:
                if train: # we add dropout to fc6 and fc7 during training
                    print('Using dropout for ' + node)
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 0.75
                else: # turn off dropout during training
                    print('Not using dropout')
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 1.0

            memory_func, memory_param = attr['kwargs']['memory']
            if 'filter_size' in memory_param:
                attr['cell'] = tnn_ConvLSTMCell
            elif 'nunits' in memory_param:
                assert(0)
                attr['cell'] = tnn_DenseRNNCell
            else:
                attr['kwargs']['memory'][1]['memory_decay'] = tau
                attr['kwargs']['memory'][1]['trainable'] = trainable_flag

        # add any non feedforward connections here: [('L2', 'L1')]
        G.add_edges_from(edges_arr)

        # initialize network to infer the shapes of all the parameters
        main.init_nodes(G, input_nodes=['conv1'], batch_size=batch_size, channel_op=channel_op)
        # unroll the network through time
        main.unroll(G, input_seq={'conv1': input_list}, ntimes=ntimes)

        outputs = {}
        # start from the final output of the model and num timesteps beyond that
        # for t in range(ntimes-NUM_TIMESTEPS, ntimes):
        #     idx = t - (ntimes - NUM_TIMESTEPS) # keys start at timepoint 0
        #    outputs[idx] = G.node['fc8']['outputs'][t]
        
        # alternatively, we return the final output of the model at the last timestep
        outputs['pred'] = G.node['fc3']['outputs'][-1]
        return outputs, params

def rnn_fc(inputs, train=True, prefix=MODEL_PREFIX, devices=DEVICES, num_gpus=NUM_GPUS, ntimes=TOTAL_TIMESTEPS, edges_arr=[], base_name='retina_tnn_fc_lstm', tau=0.0, trainable_flag=False, channel_op='concat', seed=0, cfg_final=None):

    params = OrderedDict()
    batch_size = inputs['images'].get_shape().as_list()[0]

    params['stim_type'] = stim_type
    params['train'] = train
    params['batch_size'] = batch_size

    input_images = inputs['images']
    input_images = tf.reshape(input_images, [batch_size, 40, 50, 50, 1])

    # Accepts list of length 40. [40, batch size, 50, 50, 1]
    input_list = []
    for i in range(40):
        slice_val = tf.squeeze(tf.slice(input_images, [0, i, 0, 0, 0], [-1, 1, -1, -1, -1]), axis=1)
        input_list.append(slice_val)
    input_list.append(slice_val)
    input_list.append(slice_val)
    input_list.append(slice_val)

    with tf.variable_scope("retina_tnn_fc_lstm"):
        base_name += '.json'
        print('Using base: ', base_name)
        # creates the feedforward network graph from json
        G = main.graph_from_json(base_name)

        for node, attr in G.nodes(data=True):
            if node in ['conv1', 'conv2']:
                if train: # we add dropout to fc6 and fc7 during training
                    print('Using dropout for ' + node)
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 0.75
                else: # turn off dropout during training
                    print('Not using dropout')
                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 1.0

#            if node in ['fc3']:
#                if train: # we add dropout to fc3 during training
#                    print('Using dropout for ' + node)
#                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 0.5
#               else: # turn off dropout during training
#                    print('Not using dropout')
#                    attr['kwargs']['post_memory'][1][1]['keep_prob'] = 1.0

            memory_func, memory_param = attr['kwargs']['memory']
            if 'filter_size' in memory_param:
                assert(0) # Should not be here in this specific arch
                attr['cell'] = tnn_ConvLSTMCell
            elif 'nunits' in memory_param:
                attr['cell'] = tnn_DenseRNNCell
            else:
                attr['kwargs']['memory'][1]['memory_decay'] = tau
                attr['kwargs']['memory'][1]['trainable'] = trainable_flag

        # add any non feedforward connections here: [('L2', 'L1')]
        G.add_edges_from(edges_arr)

        # initialize network to infer the shapes of all the parameters
        main.init_nodes(G, input_nodes=['conv1'], batch_size=batch_size, channel_op=channel_op)
        # unroll the network through time
        main.unroll(G, input_seq={'conv1': input_list}, ntimes=ntimes)

        outputs = {}
        # start from the final output of the model and num timesteps beyond that
        # for t in range(ntimes-NUM_TIMESTEPS, ntimes):
        #     idx = t - (ntimes - NUM_TIMESTEPS) # keys start at timepoint 0
        #    outputs[idx] = G.node['fc8']['outputs'][t]
        
        # alternatively, we return the final output of the model at the last timestep
        outputs['pred'] = G.node['fc3']['outputs'][-1]
        return outputs, params

def poisson_loss(logits, labels):
    epsilon = 1e-8
    logits = logits["pred"]
    #N = logits.get_shape().as_list()[1]
    #loss = 1.0/N * tf.reduce_sum(logits - labels * tf.log(logits + epsilon), 1)
    loss = logits - labels * tf.log(logits + epsilon)
    return loss
    
def mean_loss_with_reg(loss):
    return tf.reduce_mean(loss)# + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

def online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(np.mean(v))
    return agg_res

def loss_metric(inputs, outputs, target, **kwargs):
    sys.stdout.flush()
    metrics_dict = {}
    metrics_dict['poisson_loss'] = mean_loss_with_reg(poisson_loss(logits=outputs, labels=inputs[target]), **kwargs)
    return metrics_dict

# def pearson_correlation_py(logits, labels):
#     s_logits = np.mean(logits, axis=0)
#     s_labels = np.mean(labels, axis=0)
#     std_logits = np.std(logits, axis=0)
#     std_labels = np.std(labels, axis=0)
#     r = np.sum((logits - s_logits)*(labels - s_labels), axis=0)/(std_logits * std_labels)
#     return r

# def pearson_correlation(logits, labels):
#     return tf.py_func(pearson_correlation_py, [logits, labels], tf.float32)

# model parameters

def get_targets(inputs, outputs, target, **kwargs):
    targets_dict = {}
    targets_dict['pred'] = outputs['pred']
    targets_dict['label'] = inputs[target]
    
    return targets_dict

def my_online_agg(agg_res, res, step):
    if agg_res is None:
        agg_res = {k: [] for k in res}
    for k, v in res.items():
        agg_res[k].append(v)
    return agg_res

def get_pearson(pred, truth):
    y_hat_mu = pred - np.mean(pred, axis=0, keepdims=True)
    y_mu = truth - np.mean(truth, axis=0, keepdims=True)
    
    y_hat_std = np.std(pred, axis=0, keepdims=True)
    y_std = np.std(truth, axis=0, keepdims=True)
    corr = np.mean(y_mu * y_hat_mu, axis=0, keepdims=True)/(y_std * y_hat_std)
    return corr

def pearson_agg(results):
    for k,v in results.iteritems():
        results[k] = np.concatenate(v, axis=0)

    testcorrs = {}
    testcorrs['corr'] = get_pearson(results['pred'], results['label'])

    return testcorrs

default_params = {
    'save_params': {
        'host': 'localhost',
        'port': 27017,
        'dbname': 'deepretina_recurrence',
        'collname': stim_type,
        'exp_id': 'trainval0',

        'do_save': True,
        'save_initial_filters': True,
        'save_metrics_freq': 50,  # keeps loss from every SAVE_LOSS_FREQ steps.
        'save_valid_freq': 50,
        'save_filters_freq': 50,
        'cache_filters_freq': 50,
        # 'cache_dir': None,  # defaults to '~/.tfutils'
    },

    'load_params': {
        'do_restore': False,
        'query': None
    },

    'model_params': {
        'func': None,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    },

    'train_params': {
        'minibatch_size': MB_SIZE,
        'data_params': {
            'func': retinaTF,
            'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
            'resize': IMAGE_SIZE_RESIZE,
            'batch_size': INPUT_BATCH_SIZE,
            'file_pattern': 'train*.tfrecords',
            'n_threads': 4
        },
        'queue_params': {
            'queue_type': 'random',
            'batch_size': OUTPUT_BATCH_SIZE,
            'capacity': 11*INPUT_BATCH_SIZE,
            'min_after_dequeue': 10*INPUT_BATCH_SIZE,
            'seed': 0,
        },
        'thres_loss': float('inf'),
        'num_steps': 50 * NUM_BATCHES_PER_EPOCH,  # number of steps to train
        'validate_first': True,
    },

    'loss_params': {
        'targets': ['labels'],
        'agg_func': mean_loss_with_reg,
        'loss_per_case_func': poisson_loss
    },

    'learning_rate_params': {
        'func': tf.train.exponential_decay,
        'learning_rate': 1e-3,
        'decay_rate': 1.0, # constant learning rate
        'decay_steps': NUM_BATCHES_PER_EPOCH,
        'staircase': True
    },

    'optimizer_params': {
        'func': optimizer.ClipOptimizer,
        'optimizer_class': tf.train.AdamOptimizer,
        'clip': True,
        'trainable_names': None
    },

    'validation_params': {
        'test_loss': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 4
            },
            'targets': {
                'func': loss_metric,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': N_TEST // MB_SIZE + 1,
            'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': online_agg
        },
#          'train_loss': {
#              'data_params': {
#                  'func': retinaTF,
#                  'source_dirs': [os.path.join(DATA_PATH, 'images'), os.path.join(DATA_PATH, 'labels')],
#                  'resize': IMAGE_SIZE_RESIZE,
#                  'batch_size': INPUT_BATCH_SIZE,
#                  'file_pattern': 'train*.tfrecords',
#                  'n_threads': 4
#              },
#              'targets': {
#                  'func': loss_metric,
#                  'target': 'labels',
#              },
#              'queue_params': {
#                  'queue_type': 'fifo',
#                  'batch_size': MB_SIZE,
#                  'capacity': 11*INPUT_BATCH_SIZE,
#                  'min_after_dequeue': 10*INPUT_BATCH_SIZE,
#                  'seed': 0,
#              },
#              'num_steps': N_TRAIN // OUTPUT_BATCH_SIZE + 1,
#              'agg_func': lambda x: {k: np.mean(v) for k, v in x.items()},
#              'online_agg_func': online_agg
#          },
        'white_noise_testcorr': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join('/datasets/deepretina_data/tf_records/whitenoise', 'images'), os.path.join('/datasets/deepretina_data/tf_records/whitenoise', 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 1
            },
            'targets': {
                'func': get_targets,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': 5957 // MB_SIZE + 1,
            'agg_func': pearson_agg,   # lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': my_online_agg
        },
        'natural_scenes_testcorr': {
            'data_params': {
                'func': retinaTF,
                'source_dirs': [os.path.join('/datasets/deepretina_data/tf_records/naturalscene', 'images'), os.path.join('/datasets/deepretina_data/tf_records/naturalscene', 'labels')],
                'resize': IMAGE_SIZE_RESIZE,
                'batch_size': INPUT_BATCH_SIZE,
                'file_pattern': 'test*.tfrecords',
                'n_threads': 1
            },
            'targets': {
                'func': get_targets,
                'target': 'labels',
            },
            'queue_params': {
                'queue_type': 'fifo',
                'batch_size': MB_SIZE,
                'capacity': 11*INPUT_BATCH_SIZE,
                'min_after_dequeue': 10*INPUT_BATCH_SIZE,
                'seed': 0,
            },
            'num_steps': 5956 // MB_SIZE + 1,
            'agg_func': pearson_agg,   # lambda x: {k: np.mean(v) for k, v in x.items()},
            'online_agg_func': my_online_agg
        },
    },
    'log_device_placement': False,  # if variable placement has to be logged
}

# TEST
def train_cnn_lstm():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'cnn_lstm'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    # Set to True of starting training again
    params['load_params']['do_restore'] = False

    params['model_params'] = {
        'func': model_func,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    }

    params['learning_rate_params']['learning_rate'] = 1e-5
    base.train_from_params(**params)

# MODELS
def train_cnn_lstm_dropout():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'cnn_lstm_dropout'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    # Set to True if starting training again
    params['load_params']['do_restore'] = False

    params['model_params'] = {
        'func': convLstmDropout,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    }

    params['learning_rate_params']['learning_rate'] = 1e-5
    base.train_from_params(**params)

def train_cnn_lstm_dropout_fb():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'cnn_lstm_dropout_fb'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    # Set to True if starting training again
    params['load_params']['do_restore'] = True

    params['model_params'] = {
        'func': convLstmDropout,
        'edges_arr': [('conv2', 'conv1')],
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    }

    params['learning_rate_params']['learning_rate'] = 1e-5
    base.train_from_params(**params)

def train_cnn_lstm_dropout_no_pre_mem():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'cnn_lstm_dropout_no_pre_mem'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    # Set to True if starting training again
    params['load_params']['do_restore'] = False

    params['model_params'] = {
        'func': convLstmDropout_no_pre_mem,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    }

    params['learning_rate_params']['learning_rate'] = 1e-5
    base.train_from_params(**params)

def train_cnn_fc_lstm():
    params = copy.deepcopy(default_params)
    params['save_params']['dbname'] = 'cnn_fc_lstm'
    params['save_params']['collname'] = stim_type
    params['save_params']['exp_id'] = 'trainval0'

    # Set to True if starting training again
    params['load_params']['do_restore'] = False

    params['model_params'] = {
        'func': rnn_fc,
        'num_gpus': NUM_GPUS,
        'devices': DEVICES,
        'prefix': MODEL_PREFIX
    }

    params['learning_rate_params']['learning_rate'] = 1e-5
    base.train_from_params(**params)

if __name__ == '__main__':
#    train_cnn_lstm()

# MODELS
#    train_cnn_lstm_dropout()
    train_cnn_lstm_dropout_no_pre_mem()
#    train_cnn_fc_lstm()
#    train_cnn_lstm_dropout_fb()



