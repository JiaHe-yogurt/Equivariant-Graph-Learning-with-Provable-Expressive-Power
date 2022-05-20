import tensorflow.compat.v1 as tf
import numpy as np
import copy
from tensorflow.python.ops import bitwise_ops


def hierarchy_ops_order3(inputs,normalization='inf'):
    dim = tf.to_int32(tf.shape(inputs)[4])  # extract dimension

    ## R^n^3 --> R^n^2
    # sum
    op1=tf.reduce_sum(inputs,axis=2)  # sum at each slice
    op2=tf.reduce_sum(inputs,axis=3)  # sum of columns
    op3=tf.reduce_sum(inputs,axis=4)  # sum of rows
    # extract diagonal
    op4=tf.matrix_diag_part(inputs)
    op5=tf.matrix_diag_part(tf.transpose(inputs,(0,1,3,4,2)))
    op6=tf.matrix_diag_part(tf.transpose(inputs,(0,1,4,2,3)))

    ## R^n^3 --> R^n^3
    op16 = tf.tile(tf.expand_dims(op1, axis=2), [1, 1, dim, 1, 1])-tf.matrix_diag(op5)  # N x D x m x m
    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    op17 = tf.tile(tf.expand_dims(op2, axis=3), [1, 1, 1, dim, 1])-tf.matrix_diag(op6)  # N x D x m x m
    op18 = tf.tile(tf.expand_dims(op3, axis=4), [1, 1, 1, 1, dim])-tf.matrix_diag(op4)  # N x D x m x m
    op19 = inputs

    ## R^n^3 --> R^n
    # sum
    op7=tf.reduce_sum(op1,axis=2)  # sum of side slice
    op8=tf.reduce_sum(op2,axis=3)  # sum of front slice
    op9=tf.reduce_sum(op3,axis=2)  # sum of upper slice
    # extract diagonal
    op10=tf.matrix_diag_part(op4)
    # sum diagonal
    op11=tf.reduce_sum(op4,axis=3)
    op12=tf.reduce_sum(op5,axis=3)
    op13=tf.reduce_sum(op6,axis=3)
    ## R^n^3 --> R
    # sum
    op14=tf.reduce_sum(op7, axis=2)
    # sum diagonal
    op15=tf.reduce_sum(op10, axis=2)

    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            op1 = tf.divide(op1, float_dim)
            op2 = tf.divide(op2, float_dim)
            op3 = tf.divide(op3, float_dim)
            op7 = tf.divide(op7, float_dim)
            op8 = tf.divide(op8, float_dim)
            op9 = tf.divide(op9, float_dim)
            op11 = tf.divide(op11, float_dim)
            op12 = tf.divide(op12, float_dim)
            op13 = tf.divide(op13, float_dim)
            op14 = tf.divide(op14, float_dim)
            op15 = tf.divide(op15, float_dim)


    return [op16, op17, op18, op19], [op1,op2,op3,op4,op5,op6],[op7,op8,op9,op10,op11,op12,op13],[op14,op15]


def hierarchy_invariant_order3(name, input_depth, output_depth, inputs):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        order3, order2, order1, order0 = hierarchy_ops_order3(inputs)
        order3 = tf.cast(tf.stack(order3, axis=2), 'float32')  # N x D x B
        order2 = tf.cast(tf.stack(order2, axis=2), 'float32')  # N x D x B
        order1 = tf.cast(tf.stack(order1, axis=2), 'float32')  # N x D x B
        order0 = tf.cast(tf.stack(order0, axis=2), 'float32')  # N x D x B

        #basis_dimension3 = order3.shape[2]
        basis_dimension2 = order2.shape[2]
        basis_dimension1 = order1.shape[2]
        basis_dimension0 = order0.shape[2]

        ## initialization values for variables
        ### R^n^3

        coeffs_values3 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension3], dtype=tf.float32),tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        # define variables
        coeffs3 = tf.get_variable('coeffs3', initializer=coeffs_values3)
        output3 = tf.einsum('dsb,ndbijk->nsijk', coeffs3, order3)  # N x S x m x m
        all_bias3 = tf.get_variable('all_bias3', initializer=tf.zeros([1, output_depth, 1,1,1], dtype=tf.float32))
        output3 = output3 + all_bias3

        ### R^n^2
        coeffs_values2 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension2], dtype=tf.float32),
                                    tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        # define variables
        coeffs2 = tf.get_variable('coeffs2', initializer=coeffs_values2)
        output2 = tf.einsum('dsb,ndbij->nsij', coeffs2, order2)  # N x S x m x m

        # bias
        diag_bias = tf.get_variable('diag_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        all_bias2 = tf.get_variable('all_bias2', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        mat_diag_bias = tf.multiply(tf.expand_dims(tf.expand_dims(tf.eye(tf.to_int32(tf.shape(inputs)[3])), 0), 0),
                                    diag_bias)
        output2 = output2 + all_bias2 + mat_diag_bias


        ##### R^n
        coeffs_values1 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension1], dtype=tf.float32),tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        coeffs1 = tf.get_variable('coeffs1', initializer=coeffs_values1)
        output1 = tf.einsum('dsb,ndbi->nsi', coeffs1, order1)  # N x S x m

        #bias
        all_bias1 = tf.get_variable('all_bias1', initializer=tf.zeros([1, output_depth, 1], dtype=tf.float32))
        output1 = output1 + all_bias1

        ##### R
        coeffs_values0 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension0], dtype=tf.float32),tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        coeffs0 = tf.get_variable('coeffs0', initializer=coeffs_values0)
        output0 = tf.einsum('sdb,nsb->nd', coeffs0, order0)  # N x D


        return [output3, output2, output1, output0]


def fwl_hierarchy_ops_order2(inputs, input_depth, normalization='inf'):
    dim = tf.to_int32(tf.shape(inputs)[3])  # extract dimension

    ## R^n^2 --> R^n
    # sum
    op1=tf.reduce_sum(inputs,axis=3)  # sum of rows
    op2=tf.reduce_sum(inputs,axis=2)  # sum of columns
    # extract diagonal
    op3=tf.matrix_diag_part(inputs)

    ## R^n^2 --> R
    # sum
    op4=tf.reduce_sum(op1,axis=2)  # sum of side slice
    # sum diagonal
    op5=tf.reduce_sum(op3, axis=2)

    ## R^n^2 --> R^n^3

    op6= tf.cast(tf.reshape(tf.repeat(inputs, repeats=dim, axis=2), (-1,input_depth,dim,dim,dim)), tf.int64)
    op7= tf.cast(tf.reshape(tf.tile(tf.transpose(inputs,[0,1,3,2]), [1,1,dim,1]), (-1,input_depth,dim,dim,dim)), tf.int64)

    op10=  bitwise_ops.bitwise_and(op6, op7)
    op11= 1- bitwise_ops.bitwise_or(op6, op7)
    #op12= bitwise_ops.bitwise_and(op6, op6-op7)
    #op13= bitwise_ops.bitwise_and(op7, op7-op6)
    ones = tf.cast(tf.ones((1,1,dim,dim,dim)), tf.int64)
    op14 = bitwise_ops.bitwise_and(bitwise_ops.bitwise_xor(ones, op6), op7)
    op15 = bitwise_ops.bitwise_and(bitwise_ops.bitwise_xor(ones, op7), op6)

    ## R^n^2 --> R^n^2
    op8 = tf.tile(tf.expand_dims(op1, axis=3), [1, 1, 1, dim]) - tf.matrix_diag(op1)  # N x D x m x m
    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    op9 = tf.tile(tf.expand_dims(op2, axis=2), [1, 1, dim, 1]) - tf.matrix_diag(op2)  # N x D x m x m
    op16 = inputs

    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            op1 = tf.divide(op1, float_dim)
            op2 = tf.divide(op2, float_dim)
            op4 = tf.divide(op4, float_dim)
            op5 = tf.divide(op5, float_dim)
            op8 = tf.divide(op8, float_dim)
            op9 = tf.divide(op9, float_dim)



    return  [op10, op11, op14, op15], [op8, op9, op16], [op1,op2,op3], [op4,op5]


def fwl_hierarchy_invariant_order2(name, input_depth, output_depth, inputs):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        order3, order2, order1, order0 = fwl_hierarchy_ops_order2(inputs,  input_depth)
        order3 = tf.cast(tf.stack(order3, axis=2), 'float32')  # N x D x B
        order2 = tf.cast(tf.stack(order2, axis=2), 'float32')  # N x D x B
        order1 = tf.cast(tf.stack(order1, axis=2), 'float32')  # N x D x B
        order0 = tf.cast(tf.stack(order0, axis=2), 'float32')  # N x D x B


        basis_dimension3 = order3.shape[2]
        basis_dimension2 = order2.shape[2]
        basis_dimension1 = order1.shape[2]
        basis_dimension0 = order0.shape[2]

        # initialization values for variables

        ##### R^n^3
        coeffs_values3 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension3], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))
        coeffs3 = tf.get_variable('coeffs3', initializer=coeffs_values3)
        output3 = tf.einsum('dsb,ndbijk->nsijk', coeffs3, order3)  # N x S x m x m x m
        all_bias3 = tf.get_variable('all_bias3', initializer=tf.zeros([1, output_depth, 1,1,1], dtype=tf.float32))
        output3 = output3 + all_bias3

        ##### R^n^2
        coeffs_values2 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension2], dtype=tf.float32),
                                     tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        coeffs2 = tf.get_variable('coeffs2', initializer=coeffs_values2)
        output2 = tf.einsum('dsb,ndbij->nsij', coeffs2 , order2)  # N x S x m x m

        diag_bias = tf.get_variable('diag_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        all_bias2 = tf.get_variable('all_bias2', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        mat_diag_bias = tf.multiply(tf.expand_dims(tf.expand_dims(tf.eye(tf.to_int32(tf.shape(inputs)[3])), 0), 0),
                                    diag_bias)
        output2 = output2 + all_bias2 + mat_diag_bias


        ##### R^n
        coeffs_values1 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension1], dtype=tf.float32),tf.sqrt(2. / tf.to_float(input_depth + output_depth)))


        coeffs1 = tf.get_variable('coeffs1', initializer=coeffs_values1)
        output1 = tf.einsum('dsb,ndbi->nsi', coeffs1, order1)  # N x S x m

        all_bias1 = tf.get_variable('all_bias1', initializer=tf.zeros([1, output_depth,  1], dtype=tf.float32))
        output1 = output1 + all_bias1

        ##### R
        coeffs_values0 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension0], dtype=tf.float32), tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        coeffs0 = tf.get_variable('coeffs0', initializer=coeffs_values0)
        output0 = tf.einsum('sdb,nsb->nd', coeffs0, order0)  # N x D

        return [output3, output2, output1, output0]


def wl_hierarchy_ops_order2(inputs, normalization='inf'):
    dim = tf.to_int32(tf.shape(inputs)[3])  # extract dimension
    ## R^n^2 --> R^n
    # sum
    our_op1 = tf.reduce_sum(inputs, axis=3)  # sum of rows
    our_op2 = tf.reduce_sum(inputs, axis=2)  # sum of columns

    # extract diagonal
    our_op3 = tf.matrix_diag_part(inputs)

    ## R^n^2 --> R
    # sum
    our_op4 = tf.reduce_sum(our_op1, axis=2)  # sum of side slice
    # sum diagonal
    our_op5 = tf.reduce_sum(our_op3, axis=2)

    ## R^n^2 --> R^n^2
    op6 = tf.tile(tf.expand_dims(our_op1, axis=3), [1, 1, 1, dim])  # N x D x m x m
    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    op7 = tf.tile(tf.expand_dims(our_op2, axis=2), [1, 1, dim, 1])  # N x D x m x m
    # identity
    op8 = inputs  # N x D x m x m
    op15 = tf.tile(tf.expand_dims(tf.expand_dims(our_op4, axis=2), axis=3), [1, 1, dim, dim])  # N x D x m x m

    # sum of all ops - place sum of all entries in all entries
    if normalization is not None:
        float_dim = tf.to_float(dim)
        if normalization is 'inf':
            our_op1 = tf.divide(our_op1, float_dim)
            our_op2 = tf.divide(our_op2, float_dim)
            our_op3 = tf.divide(our_op3, float_dim)
            our_op4 = tf.divide(our_op4, float_dim)
            op6 = tf.divide(op6, float_dim)
            op7 = tf.divide(op7, float_dim)
            op15 = tf.divide(op15, float_dim ** 2)

    return [op6, op7, op8, op15], [our_op1, our_op2, our_op3], [our_op4, our_op5]


def wl_hierarchy_invariant_order2(name, input_depth, output_depth, inputs):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        order2, order1, order0 = wl_hierarchy_ops_order2(inputs)
        order2 = tf.cast(tf.stack(order2, axis=2), 'float32')  # N x D x B
        order1 = tf.cast(tf.stack(order1, axis=2), 'float32')  # N x D x B
        order0 = tf.cast(tf.stack(order0, axis=2), 'float32')  # N x D x B

        # order1 = wl_hierarchy_ops_order2(inputs)
        # order1 = tf.cast(tf.stack(order1, axis=2), 'float32')  # N x D x B
        basis_dimension2 = order2.shape[2]
        basis_dimension1 = order1.shape[2]
        basis_dimension0 = order0.shape[2]

        # initialization values for variables
        ##### R^n^2
        coeffs_values2 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension2], dtype=tf.float32),
                                     tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        coeffs2 = tf.get_variable('coeffs2', initializer=coeffs_values2)
        output2 = tf.einsum('dsb,ndbij->nsij', coeffs2, order2)  # N x S x m x m

        ##### R^n
        coeffs_values1 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension1], dtype=tf.float32),
                                     tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        coeffs1 = tf.get_variable('coeffs1', initializer=coeffs_values1)
        output1 = tf.einsum('dsb,ndbi->nsi', coeffs1, order1)  # N x S x m

        ##### R
        coeffs_values0 = tf.multiply(tf.random_normal([input_depth, output_depth, basis_dimension0], dtype=tf.float32),
                                     tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        coeffs0 = tf.get_variable('coeffs0', initializer=coeffs_values0)
        output0 = tf.einsum('sdb,nsb->nd', coeffs0, order0)  # N x D

        # bias
        diag_bias = tf.get_variable('diag_bias', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        all_bias2 = tf.get_variable('all_bias2', initializer=tf.zeros([1, output_depth, 1, 1], dtype=tf.float32))
        mat_diag_bias = tf.multiply(tf.expand_dims(tf.expand_dims(tf.eye(tf.to_int32(tf.shape(inputs)[3])), 0), 0),
                                    diag_bias)
        output2 = output2 + all_bias2 + mat_diag_bias

        all_bias1 = tf.get_variable('all_bias1', initializer=tf.zeros([1, output_depth,  1], dtype=tf.float32))
        output1 = output1 + all_bias1

        return [output2, output1, output0]




def create_convolutional_layer(name, input_depth, output_depth, convolved_nodal, num_hop):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE) as scope:
        coeffs_values1 = tf.multiply(tf.random_normal([num_hop, input_depth, output_depth], dtype=tf.float32),
                                     tf.sqrt(2. / tf.to_float(input_depth + output_depth)))

        coeffs = tf.get_variable('coeffs1', initializer=coeffs_values1)
        biases = tf.Variable(tf.constant(0.01, shape=[output_depth]))

        ## Creating the convolutional layer
        supports = []
        for i in range(num_hop):
            propagate = tf.matmul(convolved_nodal[:, i ,: , :], coeffs[i,:,:])            # N x S x #output_feature
            supports.append(propagate)
        output = tf.cast(tf.add_n(supports), 'float32')+biases
        output = tf.transpose(output, [0,2,1])

        return output













