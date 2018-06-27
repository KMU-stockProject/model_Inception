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
from random import shuffle


class Inception(object):
    def __init__(self):
        self.current_dir = os.getcwd()
        self.epoch = 1500
        self.dataPath = os.path.join(self.current_dir, 'data', 'pklData')

        self.dropout_rate = tf.placeholder(dtype=tf.float32)
        self.input1 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 5])
        self.input2 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.input3 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.input4 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.input5 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.input6 = tf.placeholder(dtype=tf.float32, shape=[None, 26, 4])
        self.target = tf.placeholder(dtype=tf.float32, shape=[None, 2])

        self.isTraining = True

        self.learningRate = 0.001
        self.past = 0.0
        self.cnt = 0

        self.best = 0.0
        self.training_best = 0.0
        self.test_best = 0.0
        self.direction_best = 0.0

        self.results = [0.0 for _ in range(4)]

        self.model()

        self.sess = None

        tf.set_random_seed(777)  # reproducibility
        print('isOK')

    def stem(self, data, size):
        data = tf.reshape(data, [-1, 26, size, 1])

        stem_conv1 = tf.layers.conv2d(inputs=data, filters=5, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        stem_dropout1 = tf.nn.dropout(stem_conv1, keep_prob=self.dropout_rate)


        stem_conv2 = tf.layers.conv2d(inputs=stem_dropout1, filters=5, kernel_size=[3, 3], padding="SAME", activation=tf.nn.relu)
        stem_dropout2 = tf.nn.dropout(stem_conv2, keep_prob=self.dropout_rate)


        stem_conv3 = tf.layers.conv2d(inputs=stem_dropout2, filters=9, kernel_size=[3, 3], padding="SAME", strides=(2, 2),
                                      activation=tf.nn.relu)
        stem_dropout3_1 = tf.nn.dropout(stem_conv3, keep_prob=self.dropout_rate)

        stem_pool3 = tf.layers.max_pooling2d(inputs=stem_dropout2, pool_size=[3, 3], padding="SAME", strides=2)
        stem_dropout3_2 = tf.nn.dropout(stem_pool3, keep_prob=self.dropout_rate)

        inter_data = tf.concat([stem_dropout3_1, stem_dropout3_2], axis=3)
        print(inter_data, stem_dropout3_1, stem_dropout3_2)
        return tf.layers.conv2d(inputs=inter_data, filters=10, kernel_size=[1, 1], padding="SAME", activation=tf.nn.relu)


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
        input1 = tf.layers.batch_normalization(self.input1, training=self.isTraining)
        input2 = tf.layers.batch_normalization(self.input2, training=self.isTraining)
        input3 = tf.layers.batch_normalization(self.input3, training=self.isTraining)
        input4 = tf.layers.batch_normalization(self.input4, training=self.isTraining)
        input5 = tf.layers.batch_normalization(self.input5, training=self.isTraining)
        input6 = tf.layers.batch_normalization(self.input6, training=self.isTraining)
        print(input1, input2, input3, input4, input5, input6)
        data1 = self.stem(input1, 5)
        data2 = self.stem(input2, 4)
        data3 = self.stem(input3, 4)
        data4 = self.stem(input4, 4)
        data5 = self.stem(input5, 4)
        data6 = self.stem(input6, 4)
        print(data1, data2, data3, data4, data5, data6)
        data = tf.concat([data1, data2, data3, data4, data5, data6], axis=2)
        data = tf.layers.batch_normalization(data, training=self.isTraining)

        output_inception_a = self.inception_a(data)
        output_reduction_a = self.reduction_a(output_inception_a)

        output_inception_b = self.inception_b(output_reduction_a)
        output_reduction_b = self.reduction_b(output_inception_b)

        output_inception_c = self.inception_c(output_reduction_b)
        print(output_reduction_b, output_inception_c)
        output = tf.layers.average_pooling2d(inputs=output_inception_c, pool_size=[2, 2], padding="VALID", strides=1)


        output = tf.reshape(output, [-1, 100])

        fnn_output1 = tf.contrib.layers.fully_connected(output, 70, activation_fn=tf.nn.relu)
        dropout1 = tf.nn.dropout(fnn_output1, keep_prob=self.dropout_rate)
        fnn_output2 = tf.contrib.layers.fully_connected(dropout1, 40, activation_fn=tf.nn.relu)
        dropout2 = tf.nn.dropout(fnn_output2, keep_prob=self.dropout_rate)
        fnn_output3 = tf.contrib.layers.fully_connected(dropout2, 15, activation_fn=tf.nn.relu)
        dropout3 = tf.nn.dropout(fnn_output3, keep_prob=self.dropout_rate)
        fnn_output4 = tf.contrib.layers.fully_connected(dropout3, 9, activation_fn=tf.nn.relu)
        dropout4 = tf.nn.dropout(fnn_output4, keep_prob=self.dropout_rate)
        self.hypothesis = tf.contrib.layers.fully_connected(dropout4, 2, activation_fn=tf.nn.softmax)


        inter_output = tf.layers.average_pooling2d(inputs=output_reduction_b, pool_size=[2, 2], padding="VALID", strides=1)
        inter_output = tf.reshape(inter_output, [-1, 80])

        fop1 = tf.contrib.layers.fully_connected(inter_output, 45, activation_fn=tf.nn.relu)
        dp1 = tf.nn.dropout(fop1, keep_prob=self.dropout_rate)
        fop2 = tf.contrib.layers.fully_connected(dp1, 22, activation_fn=tf.nn.relu)
        dp2 = tf.nn.dropout(fop2, keep_prob=self.dropout_rate)
        fop3 = tf.contrib.layers.fully_connected(dp2, 9, activation_fn=tf.nn.relu)
        dp3 = tf.nn.dropout(fop3, keep_prob=self.dropout_rate)
        self.inter_hypothesis = tf.contrib.layers.fully_connected(dp3, 2, activation_fn=tf.nn.softmax)

        ################################################################### loss/optimize
        self.cost = 0.8 * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.hypothesis, labels=self.target)) \
                    + 0.2 * tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.inter_hypothesis, labels=self.target))

        self.opt = tf.train.AdamOptimizer(learning_rate=self.learningRate).minimize(self.cost)

        tf.summary.scalar("loss", self.cost)

        ################################################################### accuracy
        self.h_argmax = tf.argmax(self.hypothesis, 1)
        self.t_argmax = tf.argmax(self.target, 1)

        correct_prediction = tf.equal(self.h_argmax, self.t_argmax)
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar("accuracy", self.acc)
        self.summary = tf.summary.merge_all()


    def get_predict(self):
        self.isTraining = False
        pass

    def get_accuracy(self, data):
        return self.sess.run([self.acc, self.h_argmax, self.t_argmax],
                             feed_dict={self.input1: data[0], self.input2: data[1],
                                        self.input3: data[2], self.input4: data[3],
                                        self.input5: data[4], self.input6: data[5],
                                        self.target: data[6], self.dropout_rate: 1.0})

    def train(self, data, dropout_rate=0.7):
        return self.sess.run([self.cost, self.opt], feed_dict={self.input1: data[0], self.input2: data[1],
                                                               self.input3: data[2], self.input4: data[3],
                                                               self.input5: data[4], self.input6: data[5],
                                                               self.target: data[6], self.dropout_rate: dropout_rate})

    def run2(self):
        self.sess = tf.Session()
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state('trainedModel')

        try:
            saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            training_acc, _, _ = self.get_accuracy(training_data)
        except:
            print("Error on loading old network weights")

    def run(self):
        model_save_path = os.path.join(self.current_dir, 'trainedModel')
        tensorboard_path = os.path.join(self.current_dir, 'tensorboard')
        os.mkdir(tensorboard_path)
        os.mkdir(model_save_path)
        data_list = os.listdir(os.path.join(self.current_dir, 'data', 'pklData', 'training'))

        # os.environ["CUDA_VISIBLE_DEVICES"] = "2"

        saver = tf.train.Saver()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(tensorboard_path)
        writer.add_graph(self.sess.graph)

        for ep in range(self.epoch):
            training_acc_sum = 0.0
            for company_index in data_list[:70]:
                fp = open(os.path.join(self.dataPath, 'training', company_index), 'rb')
                training_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()
                
                loss, opt = self.train(training_data)
                training_acc, _, _ = self.get_accuracy(training_data)

                training_acc_sum += training_acc

            training_acc_sum /= len(data_list)
            if self.past > training_acc_sum:
                if self.cnt == 3:
                    self.cnt = 0
                    self.learningRate /= 1.5
                    print('learningRate:', self.learningRate)
                else:
                    self.cnt += 1
            else:
                self.cnt = 0
            self.past = training_acc_sum

            if not (ep + 1) % 5:
                print(training_acc_sum, loss)

            # writer.add_summary(summary1, ep + 1)
            # writer.add_summary(summary, ep + 1)
    
            ########################################################################################################
            #  check accuracy
            test_acc_sum = 0.0
            for company_index in data_list[70:]:
                fp = open(os.path.join(self.dataPath, 'training', company_index), 'rb')
                test_data = [np.array(data) for data in pickle.load(fp)]
                fp.close()

                test_acc, harg, targ = self.get_accuracy(test_data)
                test_acc_sum += test_acc

            test_acc_sum /= len(data_list[70:])
            if self.best < (test_acc_sum*0.6 + training_acc_sum*0.4):
                saver.save(self.sess, os.path.join(model_save_path, 'model'))
                self.training_best = training_acc_sum
                self.test_best = test_acc_sum

                self.best = test_acc_sum * 0.6 + training_acc_sum * 0.4

                test_file = open('/home/algorithm/test/test', 'a')
                for accuracy_range in [5, 10, 15, 20, 25, 30, 35, 40]:
                    accuracy_check_list = list()
                    for accuracy_check in range(accuracy_range, len(harg), accuracy_range):
                        accuracy_check_list.append(
                            round(sum([harg[q] == targ[q] for q in range(accuracy_check - accuracy_range, accuracy_check)])
                            / accuracy_range, 3))

                    test_file.write('{}\n'.format(accuracy_check_list))
                test_file.close()

            shuffle(data_list)


            if (ep+1) % 5 is 0:
                print('accuracy_best(step{}): {}, {}'.format(ep + 1, self.training_best, self.test_best))
