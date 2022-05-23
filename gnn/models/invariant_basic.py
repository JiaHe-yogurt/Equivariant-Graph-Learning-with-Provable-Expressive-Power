from models.base_model import BaseModel
import layers.equivariant_linear as eq
import layers.layers as layers
import tensorflow.compat.v1 as tf


class invariant_basic(BaseModel):
    def __init__(self, config, data):
        super(invariant_basic, self).__init__(config)
        self.data = data
        self.build_model()
        self.init_saver()



    def build_model(self):
        # here you build the tensorflow graph of any model you want and define the loss.
        self.is_training = tf.placeholder(tf.bool)

        self.labels = tf.placeholder(tf.int32, shape=[None])

        if self.config.input_order == 3:
            self.graphs = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs[0].shape[0], None, None, None])
            net3, net2, net1, net0 = eq.hierarchy_invariant_order3('eq1',  self.data.train_graphs[0].shape[0], self.config.architecture2d[0], self.graphs)
            net3 = tf.nn.relu(net3, name='relu10' )
            net2 = tf.nn.relu(net2, name='relu11' )
            net1 = tf.nn.relu(net1, name='relu12' )
            net0 = tf.nn.relu(net0, name='relu13' )

            net2, net11, net00= eq.wl_hierarchy_invariant_order2('eq2',   self.config.architecture2d[0], self.config.architecture2d[0], net2)
            net2= tf.nn.relu(net2 )
            net11 = tf.nn.relu(net11 )
            net00 = tf.nn.relu(net00 )


            net2 = layers.diag_offdiag_maxpool(net2)
            net1 = tf.reduce_sum(tf.concat([net1, net11], axis=1), axis=2)
            net0 = tf.concat([net0, net00], axis=1)

            net = tf.concat([net3, net2, net1, net0], axis=1)


        elif self.config.input_order == 2:
            self.graphs = tf.placeholder(tf.float32, shape=[None, self.data.train_graphs[0].shape[0], None, None])
            if self.config.algo == '2fwl':
                net3. net2, net1, net0= eq.fwl_hierarchy_invariant_order2('eq1',  self.data.train_graphs[0].shape[0], self.config.architecture2d[0], self.graphs)
                net3 = tf.nn.relu(net3, name='relu10')
                net2 = tf.nn.relu(net2, name='relu11')
                net1 = tf.nn.relu(net1, name='relu12')
                net0 = tf.nn.relu(net0, name='relu13')

                net2, net11, net00= eq.wl_hierarchy_invariant_order2('eq2', self.config.architecture2d[0], self.config.architecture2d[0], net2)
                net2 = tf.nn.relu(net2, name='relu12')
                net11 = tf.nn.relu(net11, name='relu12')
                net00 = tf.nn.relu(net00, name='relu13')

                net2 = layers.diag_offdiag_maxpool(net2)
                net1 = tf.reduce_sum( tf.concat([net1, net11],axis=1), axis=2)
                net0 = tf.concat([net0, net00],axis=1)
                net = tf.concat([ net3, net2, net1, net0], axis=1)


            elif self.config.algo == '2wl':
             net2, net1, net0= eq.wl_hierarchy_invariant_order2('eq1',  self.data.train_graphs[0].shape[0], self.config.architecture2d[0], self.graphs)
             net2 = tf.nn.relu(net2, name='relu11' )
             net1 = tf.nn.relu(net1, name='relu12' )
             net0 = tf.nn.relu(net0, name='relu13' )

             net2, net11, net00 = eq.wl_hierarchy_invariant_order2('eq2',  self.config.architecture2d[0], self.config.architecture2d[0], net2)
             net2 = tf.nn.relu(net2, name='relu14' )
             net11 = tf.nn.relu(net11, name='relu12' )
             net00 = tf.nn.relu(net00, name='relu13' )

             net2 = layers.diag_offdiag_maxpool(net2)

             net1 = tf.reduce_sum(tf.concat([net1, net11], axis=1), axis=2)
             net0 = tf.concat([net0, net00], axis=1)
             net = tf.concat([net2, net1, net0], axis=1)



        net = layers.fully_connected(net, 512, "full1")
        #net = tf.layers.batch_normalization(net, training=self.is_training)
        net = layers.fully_connected(net, 256, "full2")
        #net = tf.layers.batch_normalization(net, training=self.is_training)
        if self.config.task == 'classification':
          net = layers.fully_connected(net, self.config.num_classes, "full4", activation_fn=None)
        else:
          net = tf.reshape(layers.fully_connected(net, 1, "fully3", activation_fn=None),(-1,))
        # define loss function
        with tf.name_scope("loss"):
            if self.config.task == 'classification':
               self.loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=net))
               self.correct_predictions = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(net, 1, output_type=tf.int32), self.labels), tf.int32))
               self.pred= tf.argmax(net, 1, output_type=tf.int32)
            else:
               self.loss = tf.reduce_sum(tf.losses.mean_squared_error(labels=self.labels, predictions=net))        # regression
               self.correct_predictions = tf.reduce_sum(tf.losses.absolute_difference(labels=self.labels, predictions=net))        # regression
               self.pred = net
            # get learning rate with decay every 20 epochs
        learning_rate = self.get_learning_rate(self.global_step_tensor, self.data.train_size*20)

        # choose optimizer
        if self.config.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=self.config.momentum)
        elif self.config.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate)

        # define train step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
           self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_step_tensor)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def get_learning_rate(self, global_step, decay_step):
        """
        helper method to fit learning rat
        :param global_step: current index into dataset, int
        :param decay_step: decay step, float
        :return: output: N x S x m x m tensor
        """
        learning_rate = tf.train.exponential_decay(
            self.config.learning_rate,  # Base learning rate.
            global_step*self.config.batch_size,
            decay_step,
            self.config.decay_rate,  # Decay rate.
            staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.00001)
        return learning_rate
