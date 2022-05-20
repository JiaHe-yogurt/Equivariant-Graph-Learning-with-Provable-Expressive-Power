import tensorflow.compat.v1  as tf

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
    """Helper to create an initialized Variable with weight decay.
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.
      use_xavier: bool, whether to use xavier initializer
    Returns:
      Variable Tensor
    """
    if use_xavier:
        initializer = tf.keras.initializers.glorot_normal()
    else:
        initializer = tf.truncated_normal_initializer(stddev=stddev)
    var = _variable_on_cpu(name, shape, initializer)
    if wd is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def order3_invariant(input, a2d):

    ## extract diagonal
    #dim = tf.to_int32(tf.shape(input)[4])  # extract dimension
    #num_feature =  tf.to_int32(tf.shape(input)[1])
    #tensor = tf.reshape(input, [-1, 6**3*a2d[0]])
    #return tensor
    op4=tf.matrix_diag_part(input)
    op5=tf.matrix_diag_part(tf.transpose(input,(0,1,3,4,2)))
    op6=tf.matrix_diag_part(tf.transpose(input,(0,1,4,2,3)))

    diag_1=tf.reduce_sum(op4, axis=[2,3])
    diag_2=tf.reduce_sum(op5, axis=[2,3])
    diag_3=tf.reduce_sum(op6, axis=[2,3])

    sumall=tf.reduce_sum(input, axis=[2,3,4])
    return tf.concat([diag_1, diag_2, diag_3, sumall], axis=1)

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """Helper to create a Variable stored on CPU memory.
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var



def conv2d(inputs, num_output_channels, kernel_size, scope, stride=[1, 1],
           padding='SAME', use_xavier=True,stddev=1e-3,weight_decay=0.0,activation_fn=tf.nn.relu):
    """ 2D convolution with non-linear operation.
    Args:
      inputs: 4-D tensor variable BxHxWxC
      num_output_channels: int
      kernel_size: a list of 2 ints
      scope: string
      stride: a list of 2 ints
      padding: 'SAME' or 'VALID'
      use_xavier: bool, use xavier_initializer if true
      stddev: float, stddev for truncated_normal init
      weight_decay: float
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable
    Returns:
      Variable tensor
    """
    with tf.variable_scope(scope) as sc:
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.shape[-1]
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        kernel = _variable_with_weight_decay('wedigd',
                                             shape=kernel_shape,
                                             use_xavier=use_xavier,
                                             stddev=stddev,
                                             wd=weight_decay)
        stride_h, stride_w = stride
        outputs = tf.nn.conv2d(inputs, kernel,
                               [1, stride_h, stride_w, 1],
                               padding=padding)
        biases = _variable_on_cpu('biases', [num_output_channels],
                                  tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs


def diag_offdiag_maxpool(input): #input.shape BxSxNxN


    max_diag = tf.reduce_max(tf.matrix_diag_part(input), axis=2) #BxS

    max_val = tf.reduce_max(max_diag)

    min_val = tf.reduce_max(tf.multiply(input, tf.constant(-1.)))
    val = tf.abs(max_val+min_val)
    min_mat = tf.expand_dims(tf.expand_dims(tf.matrix_diag(tf.add(tf.multiply(tf.matrix_diag_part(input[0][0]),0),val)), axis=0), axis=0)
    max_offdiag = tf.reduce_max(tf.subtract(input, min_mat), axis=[2, 3])

    return tf.concat([max_diag, max_offdiag], axis=1) #output BxSx2


def spatial_dropout(x, keep_prob, is_training, seed=1234):
    output = tf.cond(is_training, lambda: spatial_dropout_imp(x, keep_prob, seed), lambda: x)
    return output

def spatial_dropout_imp(x, keep_prob, seed=1234):
    drop = keep_prob + tf.random_uniform(shape=[1, tf.shape(x)[1], 1, 1], minval=0, maxval=1, seed=seed)
    drop = tf.floor(drop)
    return tf.divide(tf.multiply(drop, x), keep_prob)


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    activation_fn=tf.nn.relu):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """
    with tf.variable_scope(scope) as sc:
    #    num_input_units = inputs.get_shape()[-1].value
    #      initializer = tf.contrib.layers.xavier_initializer()
        num_input_units = inputs.get_shape()[-1]
        initializer = tf.keras.initializers.glorot_normal()
        weights = tf.get_variable("weights", shape=[num_input_units, num_outputs],initializer=initializer, dtype=tf.float32)

        outputs = tf.matmul(inputs, weights)
        biases = tf.get_variable('biases', [num_outputs], initializer=tf.constant_initializer(0.))

        outputs = tf.nn.bias_add(outputs, biases)

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

