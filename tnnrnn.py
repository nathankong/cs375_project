import tensorflow as tf
import sys
from tensorflow.contrib.rnn import LSTMStateTuple
from tnn.cell import *

class ConvRNNCell(object):
  """Abstract object representing an Convolutional RNN cell.
  """

  def __call__(self, inputs, state, scope=None):
    """Run this RNN cell on inputs, starting from the given state.
    """
    raise NotImplementedError("Abstract method")

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
      filled with zeros
    """
    shape = self.shape
    out_depth = self._out_depth
    zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype) 
    return zeros



class ConvLSTMCell(ConvRNNCell):
  """Conv LSTM recurrent network cell.
  """

  def __init__(self,
               shape,
               filter_size, 
               out_depth,
               use_peepholes=False,
               forget_bias=1.0,
               state_is_tuple=False, 
               activation=tf.nn.tanh,
               kernel_initializer=None,
               bias_initializer=None):
    """Initialize the Conv LSTM cell.
    Args:
      shape: int tuple thats the height and width of the cell
      filter_size: int tuple thats the height and width of the filter
      out_depth: int thats the depth of the cell 
      use_peepholes: bool, set True to enable peephole connections
      activation: Activation function of the inner states.
      forget_bias: float, The bias added to forget gates (see above).
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  If False, they are concatenated
        along the column axis.  The latter behavior will soon be deprecated.
    """
    self.shape = shape
    self.filter_size = filter_size
    self._use_peepholes = use_peepholes
    self._out_depth = out_depth 
    self._size = tf.TensorShape([self.shape[0], self.shape[1], self._out_depth])
    self._concat_size = tf.TensorShape([self.shape[0], self.shape[1], 2*self._out_depth])
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return (LSTMStateTuple(self._size, self._size)
            if self._state_is_tuple else self._concat_size)

  @property
  def output_size(self):
    return self._size

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).
    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.
    Returns:
      tensor of shape '[batch_size x shape[0] x shape[1] x out_depth]
      filled with zeros
    """
    # last dimension is replaced by 2 * out_depth = (c, h)
    shape = self.shape
    out_depth = self._out_depth
    if self._state_is_tuple:
        zeros = LSTMStateTuple(
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype),
                tf.zeros([batch_size, shape[0], shape[1], out_depth], dtype=dtype))
    else:
        zeros = tf.zeros([batch_size, shape[0], shape[1], out_depth * 2], dtype=dtype)
    return zeros

  def __call__(self, inputs, state):
    """Long-short term memory (LSTM)."""
    with tf.variable_scope(type(self).__name__):  # "ConvLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency
      if self._state_is_tuple:
          c, h = state
      else:
          c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
      
      concat = _conv_linear([inputs, h], \
                                 self.filter_size, self._out_depth * 4, True)

      # i = input_gate, j = new_input, f = forget_gate, o = output_gate
      i, j, f, o = tf.split(axis=3, num_or_size_splits=4, value=concat)

      new_c = (c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) *
                             self._activation(j))
      new_h = self._activation(new_c) * tf.nn.sigmoid(o)

      if self._state_is_tuple:
          new_state = LSTMStateTuple(new_c, new_h)
      else:
          new_state = tf.concat(axis=3, values=[new_c, new_h])
      return new_h, new_state   
                        
def _conv_linear(args, filter_size, out_depth, bias, bias_start=0.0, scope=None):
    """convolution:
        Args:
            args: 4D Tensor or list of 4D, batch x n, Tensors.
            filter_size: tuple(int height, int width) of filter
            out_depth: (int out_depth) number of features.
            bias_start: starting value to initialize the bias; 0 by default.
            scope: VariableScope for the created subgraph; defaults to "Linear".
        Returns:
            A 4D Tensor with shape [batch h w out_depth]
        Raises:
            ValueError: if some of the arguments has unspecified or wrong shape.
        """

    # Calculate the total size of arguments on dimension 1.
    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]
        
    dtype = [a.dtype for a in args][0]

    # Now the computation.
    with tf.variable_scope(scope or "Conv") as conv_scope:
        matrix = tf.get_variable("Matrix", [filter_size[0], filter_size[1], \
                                   total_arg_size_depth, out_depth], dtype=dtype)
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, \
                               strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        
        bias_term = tf.get_variable("Bias", [out_depth], dtype=dtype,\
                                      initializer=tf.constant_initializer(bias_start,\
                                                                          dtype=dtype))
    return res + bias_term
    
class tnn_ConvLSTMCell(ConvRNNCell):

    def __init__(self,
                 harbor_shape,
                 harbor=(harbor, None),
                 pre_memory=None,
                 memory=(memory, None),
                 post_memory=None,
                 input_init=(tf.zeros, None),
                 state_init=(tf.zeros, None),
                 dtype=tf.float32,
                 name=None
                 ):

        self.harbor_shape = harbor_shape
        self.harbor = harbor if harbor[1] is not None else (harbor[0], {})
        self.pre_memory = pre_memory
        self.memory = memory if memory[1] is not None else (memory[0], {})
        self.post_memory = post_memory

        self.input_init = input_init if input_init[1] is not None else (input_init[0], {})
        self.state_init = state_init if state_init[1] is not None else (state_init[0], {})

        self.dtype = dtype
        self.name = name

        self._reuse = None

        self.conv_cell = ConvLSTMCell(memory[1]['shape'], memory[1]['filter_size'], memory[1]['out_depth'])

    def __call__(self, inputs=None, state=None):
        """
        Produce outputs given inputs
        If inputs or state are None, they are initialized from scratch.
        :Kwargs:
            - inputs (list)
                A list of inputs. Inputs are combined using the harbor function
            - state
        :Returns:
            (output, state)
        """

        with tf.variable_scope(self.name, reuse=self._reuse):

            if inputs is None:
                inputs = [self.input_init[0](shape=self.harbor_shape,
                                             **self.input_init[1])]
            #print(inputs, self.harbor_shape, self.name)
            sys.stdout.flush()
            output = self.harbor[0](inputs, self.harbor_shape, self.name, reuse=self._reuse, **self.harbor[1])

            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs) # component_conv needs to know the inputs
                    else:
                       output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                bs = output.get_shape().as_list()[0]
                state = self.conv_cell.zero_state(bs, dtype = self.dtype)

            output, state = self.conv_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    if function.__name__ == "component_conv":
                       output = function(output, inputs, **kwargs)
                    else:
                       output = function(output, **kwargs)
                post_name_counter += 1
            self.output = tf.identity(tf.cast(output, self.dtype), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_shape = self.output.shape
        return self.output, state

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        # if self.state is not None:
        return self.state_shape
        # else:
        #     raise ValueError('State not initialized yet')

    @property
    def output_size(self):
        """
        Integer or TensorShape: size of outputs produced by this cell.
        """
        # if self.output is not None:
        return self.output_shape
        # else:
        #     raise ValueError('Output not initialized yet')


# LSTM cell by default, can be any dense RNN cell that subclasses RNNCell (e.g. GRUCell)
class tnn_DenseRNNCell(RNNCell):

    def __init__(self,
                 harbor_shape,
                 harbor=(harbor, None),
                 pre_memory=None,
                 memory=(memory, None),
                 post_memory=None,
                 input_init=(tf.zeros, None),
                 state_init=(tf.zeros, None),
                 dtype=tf.float32,
                 name=None
                 ):

        self.harbor_shape = harbor_shape
        self.harbor = harbor if harbor[1] is not None else (harbor[0], {})
        self.pre_memory = pre_memory
        self.memory = memory if memory[1] is not None else (memory[0], {})
        self.post_memory = post_memory

        self.input_init = input_init if input_init[1] is not None else (input_init[0], {})
        self.state_init = state_init if state_init[1] is not None else (state_init[0], {})

        self.dtype = dtype
        self.name = name

        self._reuse = None

        cell_type = 'LSTMCell'
        if 'type' in memory[1]:
            cell_type = memory[1]['type']

        #self.lstm_cell = getattr(rnn, cell_type)(memory[1]['nunits'])
        self.lstm_cell = getattr(tf.contrib.rnn, cell_type)(memory[1]['nunits'])


    def __call__(self, inputs=None, state=None):
        """
        Produce outputs given inputs
        If inputs or state are None, they are initialized from scratch.
        :Kwargs:
            - inputs (list)
                A list of inputs. Inputs are combined using the harbor function
            - state
        :Returns:
            (output, state)
        """

        with tf.variable_scope(self.name, reuse=self._reuse):

            if inputs is None:
                inputs = [self.input_init[0](shape=self.harbor_shape,
                                             **self.input_init[1])]
            output = self.harbor[0](inputs, self.harbor_shape, self.name, reuse=self._reuse, **self.harbor[1])

            pre_name_counter = 0
            for function, kwargs in self.pre_memory:
                with tf.variable_scope("pre_" + str(pre_name_counter), reuse=self._reuse):
                    output = function(output, **kwargs)
                pre_name_counter += 1

            if state is None:
                state = self.lstm_cell.zero_state(output.get_shape().as_list()[0], dtype = self.dtype)

            output, state = self.lstm_cell(output, state)
            self.state = tf.identity(state, name='state')

            post_name_counter = 0
            for function, kwargs in self.post_memory:
                with tf.variable_scope("post_" + str(post_name_counter), reuse=self._reuse):
                    output = function(output, **kwargs)
                post_name_counter += 1
            self.output = tf.identity(tf.cast(output, self.dtype), name='output')

            self._reuse = True

        self.state_shape = self.state.shape
        self.output_shape = self.output.shape
        return self.output, state

    @property
    def state_size(self):
        """
        Size(s) of state(s) used by this cell.
        It can be represented by an Integer, a TensorShape or a tuple of Integers
        or TensorShapes.
        """
        # if self.state is not None:
        return self.state_shape
        # else:
        #     raise ValueError('State not initialized yet')

    @property
    def output_size(self):
        """
        Integer or TensorShape: size of outputs produced by this cell.
        """
        # if self.output is not None:
        return self.output_shape
        # else:
        #     raise ValueError('Output not initialized yet')
        
