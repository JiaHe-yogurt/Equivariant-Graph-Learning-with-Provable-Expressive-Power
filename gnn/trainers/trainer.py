from trainers.base_train import BaseTrain
import tensorflow.compat.v1 as tf
from tqdm import tqdm
import numpy as np
from Utils import doc_utils
import time

class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config):
        super(Trainer, self).__init__(sess, model, config, data)
        self.request_from_model = self.model.correct_predictions
        self.best_test_loss = np.inf
        self.best_epoch = -1
        # load the model from the latest checkpoint if exist
        self.model.load(self.sess)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs, 1):
            # train epoch
            stt=time.time()
            train_acc, train_loss = self.train_epoch(cur_epoch)
            end=time.time()
            print('train time at'+str(cur_epoch), end-stt)
            self.sess.run(self.model.increment_cur_epoch_tensor)
            # validation step
            if self.config.val_exist:
                test_acc, test_loss, self.pred, self.true = self.test(cur_epoch)
                # document results
                doc_utils.write_to_file_doc(train_acc, train_loss, test_acc, test_loss, cur_epoch, self.config)
        if self.config.val_exist:
            # creates plots for accuracy and loss during training
            doc_utils.create_experiment_results_plot(self.config.exp_name, "accuracy", self.config.summary_dir)
            doc_utils.create_experiment_results_plot(self.config.exp_name, "loss", self.config.summary_dir, log=True)
        return self.test(cur_epoch)

    def train_epoch(self, num_epoch=None):
        """
       implement the logic of epoch:
       -loop on the number of iterations in the config and call the train step

        Train one epoch
        :param epoch: cur epoch number
        :return accuracy and loss on train set
        """
        # initialize dataset
        self.data_loader.initialize(is_train=True)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.num_iterations_train), total=self.data_loader.num_iterations_train,
                  desc="epoch-{}-".format(num_epoch))

        total_loss = 0.
        total_correct = 0.

        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            loss, correct = self.train_step()
            # update results from train_step func
            total_loss += loss
            total_correct += correct

        # save model
        if num_epoch % self.config.save_rate == 0:
             self.model.save(self.sess)

        loss_per_epoch = total_loss / self.data_loader.train_size
        acc_per_epoch = total_correct / self.data_loader.train_size
        print(""" Epoch-{}  loss:{:.4f} -- acc:{:.4f}""".format(num_epoch, loss_per_epoch, acc_per_epoch))

        tt.close()
        return acc_per_epoch, loss_per_epoch

    def train_step(self):
        """
       implement the logic of the train step
       - run the tensorflow session
       - :return any accuracy and loss on current batch
       """

        graphs, labels = self.data_loader.next_batch()
        _, loss, correct = self.sess.run([self.model.train_op, self.model.loss, self.model.correct_predictions],
                                         feed_dict={self.model.graphs: graphs, self.model.labels: labels,
                                                    self.model.is_training: True})

        return loss, correct

    def test(self, epoch, load_best_model=False):
        # initialize dataset
        if load_best_model:
            self.model.load(self.sess)

        self.data_loader.initialize(is_train=False)

        # initialize tqdm
        tt = tqdm(range(self.data_loader.val_size), total=self.data_loader.val_size,
                  desc="Val-{}-".format(epoch))

        total_loss = 0.
        total_correct = 0.
        prediction = []
        true_labels = []
        # Iterate over batches
        for cur_it in tt:
            # One Train step on the current batch
            graph, label = self.data_loader.next_batch()
            if self.config.dataset_name == 'subgraph' and len(graph.shape)==3:
                graph=np.expand_dims(graph, 0)
            label = np.expand_dims(label, 0)
            pred, loss, correct = self.sess.run([self.model.pred, self.model.loss, self.model.correct_predictions],
                                                feed_dict={self.model.graphs: graph, self.model.labels: label,
                                                           self.model.is_training: False})
            # update metrics returned from train_step func
            prediction.append(pred)
            true_labels.append(label)
            total_loss += loss
            total_correct += correct

        test_loss = total_loss / self.data_loader.val_size
        test_acc = total_correct / self.data_loader.val_size

        if test_loss < self.best_test_loss:
            self.best_test_loss = test_loss
            print("New best test score achieved.")
            self.model.save(self.sess)
            self.best_epoch = epoch

        print("""
        test-{}  loss:{:.4f} -- acc:{:.4f}
        """.format(epoch, test_loss, test_acc))

        tt.close()
        return test_acc, test_loss, np.array(prediction), np.array(true_labels)

