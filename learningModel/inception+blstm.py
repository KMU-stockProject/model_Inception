"""
	Deep learning project for predicting stock trend with tensorflow.
	~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	Class file for session.

	:copyright: Hwang.S.J.
"""
import numpy
import os
import tensorflow as tf
import pickle
import numpy as np
import random


class Inception(object):
    def __init__(self, index):
        self.index = index
        self.current_dir = os.getcwd()
        self.epoch = 1500
        self.dataPath = os.path.join(self.current_dir, 'data', 'pklData')

        self.input1 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 5])
        self.input2 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.input3 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.input4 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.input5 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.input6 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, 4])

        self.isTraining = True

        self.learningRate = 0.0002
        self.past = 0.0
        self.cnt = 0

        self.best = 0.0
        self.training_best = 0.0
        self.test_best = 0.0
        self.direction_best = 0.0

        self.br = 0.0
        self.ed = 0.0
        self.eu = 0.0
        self.bl = 0.0

        self.model()
        self.costCheck()
        self.optimizer()
        self.accuracy()

        tf.set_random_seed(777)  # reproducibility
        print('isOK')

    def lstmCell(self):
        return tf.contrib.rnn.BasicLSTMCell(num_units=78, activation=tf.tanh)

    def blstm(self, data):
        output, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([self.lstmCell() for _ in range(15)],
                                                                      [self.lstmCell() for _ in range(15)],
                                                                      data, dtype=tf.float32)
        return output

    def stem(self, data, final_padding='valid'):
        # step1
        stem_conv1 = tf.layers.conv2d(inputs=data, filters=5, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        stem_conv2 = tf.layers.conv2d(inputs=stem_conv1, filters=5, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)

        stem_conv3 = tf.layers.conv2d(inputs=stem_conv2, filters=9, kernel_size=[3, 3], padding=final_padding, strides=(2, 2),
                                      activation=tf.nn.relu)
        stem_pool3 = tf.layers.max_pooling2d(inputs=stem_conv2, pool_size=[3, 3], padding=final_padding, strides=2)

        inter_data1 = tf.concat([stem_conv3, stem_pool3], axis=3)

        # step2
        stem_conv2_1_1 = tf.layers.conv2d(inputs=inter_data1, filters=10, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)
        stem_conv2_1_2 = tf.layers.conv2d(inputs=stem_conv2_1_1, filters=12, kernel_size=[3, 3], padding=final_padding, activation=tf.nn.relu)

        stem_conv2_2_1 = tf.layers.conv2d(inputs=inter_data1, filters=10, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)
        stem_conv2_2_2 = tf.layers.conv2d(inputs=stem_conv2_2_1, filters=10, kernel_size=[3, 1], padding="SAME", activation=tf.nn.relu)
        stem_conv2_2_3 = tf.layers.conv2d(inputs=stem_conv2_2_2, filters=10, kernel_size=[1, 3], padding="SAME", activation=tf.nn.relu)
        stem_conv2_2_4 = tf.layers.conv2d(inputs=stem_conv2_2_3, filters=12, kernel_size=[3, 3], padding=final_padding, activation=tf.nn.relu)

        inter_data2 = tf.concat([stem_conv2_1_2, stem_conv2_2_4], axis=3)

        return tf.layers.conv2d(inputs=inter_data2, filters=10, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)


    def inception_a(self, data):
        layer1_1 = tf.layers.average_pooling2d(inputs=data, pool_size=[3, 3], padding="SAME", strides=1)
        layer1_2 = tf.layers.conv2d(inputs=layer1_1, filters=14, kernel_size=[1, 1], padding="SAME",
                                    activation=tf.nn.relu)

        ######################################################################
        layer2_1 = tf.layers.conv2d(inputs=data, filters=14, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)

        ######################################################################
        layer3_1 = tf.layers.conv2d(inputs=data, filters=11, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)
        layer3_2 = tf.layers.conv2d(inputs=layer3_1, filters=14, kernel_size=[3, 3], padding="SAME",
                                    activation=tf.nn.relu)

        ######################################################################
        layer4_1 = tf.layers.conv2d(inputs=data, filters=11, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)
        layer4_2 = tf.layers.conv2d(inputs=layer4_1, filters=14, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        layer4_3 = tf.layers.conv2d(inputs=layer4_2, filters=14, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)

        ######################################################################
        inter_data = tf.concat([layer1_2, layer2_1, layer3_2, layer4_3], axis=3)
        return tf.layers.conv2d(inputs=inter_data, filters=28, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)

    def inception_b(self, data):
        layer1_1 = tf.layers.average_pooling2d(inputs=data, pool_size=[3, 3], padding="SAME", strides=1)
        layer1_2 = tf.layers.conv2d(inputs=layer1_1, filters=22, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)

        ######################################################################
        layer2_1 = tf.layers.conv2d(inputs=data, filters=56, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)

        ######################################################################
        layer3_1 = tf.layers.conv2d(inputs=data, filters=33, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)
        layer3_2 = tf.layers.conv2d(inputs=layer3_1, filters=38, kernel_size=[1, 7], padding="SAME", activation=tf.nn.relu)
        layer3_3 = tf.layers.conv2d(inputs=layer3_2, filters=44, kernel_size=[1, 7], padding="SAME", activation=tf.nn.relu)

        ######################################################################
        layer4_1 = tf.layers.conv2d(inputs=data, filters=33, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)
        layer4_2 = tf.layers.conv2d(inputs=layer4_1, filters=33, kernel_size=[1, 7], padding="SAME", activation=tf.nn.relu)
        layer4_3 = tf.layers.conv2d(inputs=layer4_2, filters=38, kernel_size=[7, 1], padding="SAME", activation=tf.nn.relu)
        layer4_4 = tf.layers.conv2d(inputs=layer4_3, filters=38, kernel_size=[1, 7], padding="SAME", activation=tf.nn.relu)
        layer4_5 = tf.layers.conv2d(inputs=layer4_4, filters=44, kernel_size=[7, 1], padding="SAME", activation=tf.nn.relu)

        ######################################################################
        inter_data = tf.concat([layer1_2, layer2_1, layer3_3, layer4_5], axis=3)
        return tf.layers.conv2d(inputs=inter_data, filters=70, kernel_size=[1, 1], padding="SAME",
                                activation=tf.nn.relu)


    def inception_c(self, data):
        layer1_1 = tf.layers.average_pooling2d(inputs=data, pool_size=[3, 3], padding="SAME", strides=1)
        layer1_2 = tf.layers.conv2d(inputs=layer1_1, filters=44, kernel_size=[1, 1], padding="SAME",activation=tf.nn.relu)

        ######################################################################
        layer2_1 = tf.layers.conv2d(inputs=data, filters=44, kernel_size=[1, 1], padding="SAME",
                                    activation=tf.nn.relu)

        ######################################################################
        layer3_1 = tf.layers.conv2d(inputs=data, filters=56, kernel_size=[1, 1], padding="SAME",
                                    activation=tf.nn.relu)
        layer3_2_1 = tf.layers.conv2d(inputs=layer3_1, filters=44, kernel_size=[1, 3], padding="SAME",
                                    activation=tf.nn.relu)
        layer3_2_2 = tf.layers.conv2d(inputs=layer3_1, filters=44, kernel_size=[3, 1], padding="SAME",
                                      activation=tf.nn.relu)

        ######################################################################
        layer4_1 = tf.layers.conv2d(inputs=data, filters=56, kernel_size=[1, 1], padding="SAME",
                                    activation=tf.nn.relu)
        layer4_2 = tf.layers.conv2d(inputs=layer4_1, filters=76, kernel_size=[1, 3], padding="SAME",
                                      activation=tf.nn.relu)
        layer4_3 = tf.layers.conv2d(inputs=layer4_2, filters=88, kernel_size=[3, 1], padding="SAME",
                                      activation=tf.nn.relu)
        layer4_4_1 = tf.layers.conv2d(inputs=layer4_3, filters=44, kernel_size=[1, 3], padding="SAME",
                                      activation=tf.nn.relu)
        layer4_4_2 = tf.layers.conv2d(inputs=layer4_3, filters=44, kernel_size=[3, 1], padding="SAME",
                                      activation=tf.nn.relu)

        ######################################################################
        inter_data = tf.concat([layer1_2, layer2_1, layer3_2_1, layer3_2_2, layer4_4_1, layer4_4_2], axis=3)
        return tf.layers.conv2d(inputs=inter_data, filters=100, kernel_size=[1, 1], padding="SAME",
                                activation=tf.nn.relu)

    def reduction_a(self, data):
        layer1_1 = tf.layers.max_pooling2d(inputs=data, pool_size=[3, 3], padding="VALID", strides=2)

        ######################################################################
        layer2_1 = tf.layers.conv2d(inputs=data, filters=56, kernel_size=[3, 3], padding="VALID", strides=(2, 2),
                                    activation=tf.nn.relu)

        ######################################################################
        layer3_1 = tf.layers.conv2d(inputs=data, filters=33, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)
        layer3_2 = tf.layers.conv2d(inputs=layer3_1, filters=38, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        layer3_3 = tf.layers.conv2d(inputs=layer3_2, filters=44, kernel_size=[3, 3], padding="VALID", strides=(2, 2), activation=tf.nn.relu)

        ######################################################################
        inter_data = tf.concat([layer1_1, layer2_1, layer3_3], axis=3)
        return tf.layers.conv2d(inputs=inter_data, filters=60, kernel_size=[1, 1], padding="SAME",
                                activation=tf.nn.relu)

    def reduction_b(self, data):
        layer1_1 = tf.layers.max_pooling2d(inputs=data, pool_size=[3, 3], padding="VALID", strides=2)

        ######################################################################
        layer2_1 = tf.layers.conv2d(inputs=data, filters=33, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)
        layer2_2 = tf.layers.conv2d(inputs=layer2_1, filters=33, kernel_size=[3, 3], padding="VALID", strides=(2, 2), activation=tf.nn.relu)

        ######################################################################
        layer3_1 = tf.layers.conv2d(inputs=data, filters=44, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)
        layer3_2 = tf.layers.conv2d(inputs=layer3_1, filters=44, kernel_size=[1, 7], padding="SAME", activation=tf.nn.relu)
        layer3_3 = tf.layers.conv2d(inputs=layer3_2, filters=55, kernel_size=[7, 1], padding="SAME", activation=tf.nn.relu)
        layer3_4 = tf.layers.conv2d(inputs=layer3_3, filters=55, kernel_size=[3, 3], padding="VALID", strides=(2, 2), activation=tf.nn.relu)

        ######################################################################
        inter_data = tf.concat([layer1_1, layer2_2, layer3_4], axis=3)
        return tf.layers.conv2d(inputs=inter_data, filters=80, kernel_size=[1, 1], padding="SAME",
                                activation=tf.nn.relu)


    def model(self):
        # data normalize
        input1 = tf.layers.batch_normalization(self.input1, training=self.isTraining)
        input2 = tf.layers.batch_normalization(self.input2, training=self.isTraining)
        input3 = tf.layers.batch_normalization(self.input3, training=self.isTraining)
        input4 = tf.layers.batch_normalization(self.input4, training=self.isTraining)
        input5 = tf.layers.batch_normalization(self.input5, training=self.isTraining)
        input6 = tf.layers.batch_normalization(self.input6, training=self.isTraining)

        #####################################################################################
        # blstm
        with tf.variable_scope('blstm1'):
            data1 = self.blstm(input1)
        with tf.variable_scope('blstm2'):
            data2 = self.blstm(input2)
        with tf.variable_scope('blstm3'):
            data3 = self.blstm(input3)
        with tf.variable_scope('blstm4'):
            data4 = self.blstm(input4)
        with tf.variable_scope('blstm5'):
            data5 = self.blstm(input5)
        with tf.variable_scope('blstm6'):
            data6 = self.blstm(input6)
        print(data1, data2, data3, data4, data5, data6)

        #####################################################################################
        # stemming
        with tf.variable_scope('stem1'):
            data1 = tf.reshape(data1, [-1, 26, 156, 1])
            data1 = self.stem(data1, 'same')
        with tf.variable_scope('stem2'):
            data2 = tf.reshape(data2, [-1, 26, 156, 1])
            data2 = self.stem(data2, 'same')
        with tf.variable_scope('stem3'):
            data3 = tf.reshape(data3, [-1, 26, 156, 1])
            data3 = self.stem(data3, 'same')
        with tf.variable_scope('stem4'):
            data4 = tf.reshape(data4, [-1, 26, 156, 1])
            data4 = self.stem(data4, 'same')
        with tf.variable_scope('stem5'):
            data5 = tf.reshape(data5, [-1, 26, 156, 1])
            data5 = self.stem(data5, 'same')
        with tf.variable_scope('stem6'):
            data6 = tf.reshape(data6, [-1, 26, 156, 1])
            data6 = self.stem(data6, 'same')
        print(data1, data2, data3, data4, data5, data6)

        #####################################################################################
        data = tf.concat([data1, data2, data3, data4, data5, data6], axis=1)
        data = tf.layers.batch_normalization(data, training=self.isTraining)
        print(data)

        #####################################################################################
        data = self.stem(data, 'valid')
        print(data, '!!!')

        #####################################################################################
        output_inception_a = self.inception_a(data)
        output_inception_a = self.inception_a(output_inception_a)
        output_reduction_a = self.reduction_a(output_inception_a)

        output_inception_b = self.inception_b(output_reduction_a)
        output_inception_b = self.inception_b(output_inception_b)
        output_inception_b = self.inception_b(output_inception_b)
        output_reduction_b = self.reduction_b(output_inception_b)

        output_inception_c = self.inception_c(output_reduction_b)
        output_inception_c = self.inception_c(output_inception_c)
        print(output_reduction_b, output_inception_c, 123123)
        output = tf.layers.average_pooling2d(inputs=output_inception_c, pool_size=[8, 8], padding="VALID", strides=1)

        output = tf.reshape(output, [-1, 100])

        fnn_output1 = tf.contrib.layers.fully_connected(output, 70, activation_fn=tf.nn.relu)
        dropout1 = tf.contrib.layers.dropout(fnn_output1, keep_prob=0.7, is_training=self.isTraining)
        fnn_output2 = tf.contrib.layers.fully_connected(dropout1, 40, activation_fn=tf.nn.relu)
        dropout2 = tf.contrib.layers.dropout(fnn_output2, keep_prob=0.7, is_training=self.isTraining)
        fnn_output3 = tf.contrib.layers.fully_connected(dropout2, 15, activation_fn=tf.nn.relu)
        dropout3 = tf.contrib.layers.dropout(fnn_output3, keep_prob=0.7, is_training=self.isTraining)
        fnn_output4 = tf.contrib.layers.fully_connected(dropout3, 9, activation_fn=tf.nn.relu)
        dropout4 = tf.contrib.layers.dropout(fnn_output4, keep_prob=0.7, is_training=self.isTraining)
        self.hypothesis = tf.contrib.layers.fully_connected(dropout4, 4, activation_fn=tf.nn.softmax)


        inter_output = tf.layers.average_pooling2d(inputs=output_reduction_b, pool_size=[8, 8], padding="VALID", strides=1)
        inter_output = tf.reshape(inter_output, [-1, 80])

        fop1 = tf.contrib.layers.fully_connected(inter_output, 45, activation_fn=tf.nn.relu)
        dp1 = tf.contrib.layers.dropout(fop1, keep_prob=0.7, is_training=self.isTraining)
        fop2 = tf.contrib.layers.fully_connected(dp1, 22, activation_fn=tf.nn.relu)
        dp2 = tf.contrib.layers.dropout(fop2, keep_prob=0.7, is_training=self.isTraining)
        fop3 = tf.contrib.layers.fully_connected(dp2, 9, activation_fn=tf.nn.relu)
        dp3 = tf.contrib.layers.dropout(fop3, keep_prob=0.7, is_training=self.isTraining)
        self.inter_hypothesis = tf.contrib.layers.fully_connected(dp3, 4, activation_fn=tf.nn.softmax)


    def costCheck(self):
        self.cost = 0.75*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=self.target))\
                    + 0.25*tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.inter_hypothesis, labels=self.target))
        tf.summary.scalar("loss", self.cost)

    def optimizer(self):
        self.opt = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)
        self.summary1 = tf.summary.merge_all()

    def accuracy(self):
        self.h_argmax = tf.argmax(self.hypothesis, 1)
        self.t_argmax = tf.argmax(self.target, 1)

        correct_prediction = tf.equal(self.h_argmax, self.t_argmax)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("accuracy", self.acc)
        self.summary2 = tf.summary.merge_all()


    def run(self):
        model_save_path = os.path.join(self.current_dir, 'trainedModel', os.path.splitext(self.index)[0])
        tensorboard_path = os.path.join(self.current_dir, 'tensorboard', os.path.splitext(self.index)[0])
        os.mkdir(tensorboard_path)
        os.mkdir(model_save_path)
        fff = open('result.txt', 'a')

        os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        session = tf.Session()
        saver = tf.train.Saver()

        with session as sess:
            sess.run(tf.global_variables_initializer())

            writer = tf.summary.FileWriter(tensorboard_path)
            writer.add_graph(sess.graph)

            for ep in range(self.epoch):
                self.isTraining = True
                fp = open(os.path.join(self.dataPath, 'training', self.index), 'rb')
                data = pickle.load(fp)
                s, cma, vma, cwma, vwma, l, t = data
                fp.close()

                opt, loss, summary1, acc = sess.run([self.opt, self.cost, self.summary1, self.acc],
                                                              feed_dict={self.input1: np.array(s[:5]), self.input2: np.array(cma[:5]),
                                                                         self.input3: np.array(vma[:5]), self.input4: np.array(cwma[:5]),
                                                                         self.input5: np.array(vwma[:5]), self.input6: np.array(l[:5]),
                                                                         self.target: np.array(t[:5])})

                training_acc, summary2 = sess.run([self.acc, self.summary2],
                                                  feed_dict={self.input1: np.array(s), self.input2: np.array(cma),
                                                             self.input3: np.array(vma), self.input4: np.array(cwma),
                                                             self.input5: np.array(vwma), self.input6: np.array(l),
                                                             self.target: np.array(t)})

                if self.past > training_acc:
                    if self.cnt == 2:
                        self.cnt = 0
                        self.learningRate /= 1.5
                        print('learningRate:', self.learningRate)
                    else:
                        self.cnt += 1
                else:
                    self.cnt = 0
                self.past = training_acc

                if not (ep + 1) % 5:
                    print(training_acc)

                writer.add_summary(summary1, ep + 1)
                writer.add_summary(summary2, ep + 1)


                ########################################################################################################
                #  check accuracy
                self.isTraining = False
                direction_acc = br_acc = ed_acc = eu_acc = bl_acc = 0.0

                fp = open(os.path.join(self.dataPath, 'test', self.index), 'rb')
                s, cma, vma, cwma, vwma, l, t = pickle.load(fp)
                fp.close()

                acc, harg, targ = sess.run([self.acc, self.h_argmax, self.t_argmax],
                                           feed_dict={self.input1: np.array(s), self.input2: np.array(cma),
                                                      self.input3: np.array(vma), self.input4: np.array(cwma),
                                                      self.input5: np.array(vwma), self.input6: np.array(l),
                                                      self.target: np.array(t)})

                unique, counts = numpy.unique(targ, return_counts=True)
                part_size = dict(zip(unique, counts))

                tempsum = br = ed = eu = bl = 0.0
                for j in range(len(harg)):
                    if (harg[j] < 2 and targ[j] < 2) or (harg[j] >= 2 and targ[j] >= 2):
                        tempsum += 1
                    if harg[j] == targ[j]:
                        if harg[j] == 0:
                            br += 1
                        elif harg[j] == 1:
                            ed += 1
                        elif harg[j] == 2:
                            eu += 1
                        else:
                            bl += 1

                direction_acc += tempsum / len(harg) # direction accuracy
                br_acc += br / part_size[0]   # br accuracy
                ed_acc += ed / part_size[1]   # ed accuracy
                eu_acc += eu / part_size[2]   # eu accuracy
                bl_acc += bl / part_size[3]   # bl accuracy


                if self.best < (acc*0.6 + training_acc*0.4):
                    saver.save(sess, os.path.join(model_save_path, 'model'))
                    self.direction_best = direction_acc
                    self.training_best = training_acc
                    self.test_best = acc
                    self.best = (acc*0.6 + training_acc*0.4)

                    self.br = br_acc
                    self.ed = ed_acc
                    self.eu = eu_acc
                    self.bl = bl_acc


                if (ep+1) % 5 is 0:
                    print('accuracy(step{}): {} / best: {}, {}'.format(ep + 1, acc, self.training_best, self.test_best))
                    print('direction accuracy(step{}): {} / best: {}'.format(ep + 1, direction_acc, self.direction_best))
                    print('br: {} / ed: {} / eu: {} / bl: {}'.format(self.br, self.ed, self.eu, self.bl))


        fff.write('acc: {} / direction_acc: {}\n'.format(self.test_best, self.direction_best))
        fff.write('br: {} / ed: {} / eu: {} / bl: {}\n\n'.format(self.br, self.ed, self.eu, self.bl))
        fff.close()
