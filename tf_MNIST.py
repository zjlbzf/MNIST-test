#!/usr/bin/python3.7
# -*- coding: utf-8 -*-

import datetime
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.client import timeline
import uuid
import threading
from sklearn.model_selection import KFold


flags = tf.app.flags
# [1] System
try:
    current_dir_path = os.path.split(os.path.realpath(__file__))[0]
except:
    current_dir_path = os.path.split(os.path.realpath("__file__"))[0]


def check_files_exist(dir_path):
    if os.path.exists(dir_path) == False:
        os.mkdir(dir_path)


for i in ["raw_data", "model", "log", "submit", "profile"]:
    check_files_exist(os.path.join(current_dir_path, i))

flags.DEFINE_integer("buffer_size", 20, "buffer_size")
flags.DEFINE_integer("prefetch_buffer_size", 2, "prefetch_buffer_size")

flags.DEFINE_string("current_dir_path", current_dir_path, "")

# [2] Dataset
flags.DEFINE_string("train_file_path", os.path.join(
    current_dir_path, "raw_data", "train.csv"), "train_file_path")

flags.DEFINE_list("input_shape", [28, 28, 1], "input_shape")

flags.DEFINE_integer("folds", 10, "k-flods")


# [3] Build model


# [4] Train
flags.DEFINE_integer("batch_size", 32, "")
flags.DEFINE_integer("num_parallel_calls", 8, "")
flags.DEFINE_float("learning_rate", 0.0005, "")
flags.DEFINE_float("beta_1", 0.9, "")
flags.DEFINE_float("beta_2", 0.99, "")
flags.DEFINE_float("epsilon", 1e-8, "")
flags.DEFINE_bool("decay", False, "")
flags.DEFINE_integer("epochs", 20, "")
flags.DEFINE_integer("steps_per_epoch", 100, "default is 100")

# [5] Output
flags.DEFINE_string("record_type", "record_parameters", "")

# 5.1 predict
flags.DEFINE_string("predict_x_path", os.path.join(
    current_dir_path, "raw_data", "test.csv"), "")
flags.DEFINE_integer("predict_x_batch_size", 128, "")
flags.DEFINE_integer("predict_x_steps_per_epoch", 1, "")
flags.DEFINE_string("predict_label_output_path", os.path.join(
    current_dir_path, "submit", "default_submit.csv"), "")


# 5.2 tensorborad logdir
flags.DEFINE_string("log_dir_path", " ", "log_dir_path")
flags.DEFINE_string("profile_dir_path", os.path.join(
    current_dir_path, "profile"), "profile_dir_path")

# 5.3 model save path
flags.DEFINE_string("model_save_path", " ", "")


# [6]定义分布式参数
# 参数服务器parameter server节点
flags.DEFINE_string('ps_hosts', '127.0.0.1:22221',
                    'Comma-separated list of hostname:port pairs')
# 两个worker节点
flags.DEFINE_string('worker_hosts', '127.0.0.1:22221',
                    'Comma-separated list of hostname:port pairs')
# 设置job name参数
flags.DEFINE_string('job_name', "worker", 'job name: worker or ps')
# 设置任务的索引
flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
# 选择异步并行，同步并行
flags.DEFINE_integer("issync", None, "是否采用分布式的同步模式，1表示同步模式，0表示异步模式")
flags.DEFINE_string('server_target', "None", ' ')

flags.DEFINE_integer('num_cpu_core', 8, ' ')
FLAGS = flags.FLAGS


par_str = "lr_%g_b1_%g_b2_%g_bsize_%g-%s" % (
    FLAGS.learning_rate, FLAGS.beta_1, FLAGS.beta_2, FLAGS.batch_size, str(uuid.uuid1()))
FLAGS.model_save_path = os.path.join(current_dir_path, "model", par_str)
FLAGS.predict_label_output_path = os.path.join(
    current_dir_path, "submit", par_str+".csv")
FLAGS.log_dir_path = os.path.join(current_dir_path, "log", "train", par_str)
# FLAGS.predict_label_output_path=os.path.join(current_dir_path, "submit", par_str".h5")

server = tf.train.Server.create_local_server()
PREDICT_ON = False


def Timer(func):
    def newFunc(*args, **args2):
        t1 = datetime.datetime.now()
        back = func(*args, **args2)
        print(" This function【{}】cost time:{} \n".format(
            func.__name__, datetime.datetime.now()-t1))
        return back
    return newFunc


def main(__):
    """
    1 Graph build
        1.1 data 
        1.2 model
        1.3 op 
    2.Session run
        2.1 sess confg
        2.2 run train
        2.3 run predict 
    """
    print(f"Tensorflow Version is {tf.__version__}")

    # ------------[Part 1] Graph build ----------------
    graph = tf.Graph()
    with graph.as_default():

        # with tf.device()
        # ---------****[train_data+val data]****---------
        #  1 E

        with open(FLAGS.train_file_path, "r") as f:
            for counts, __ in enumerate(f):
                pass
        dataset = tf.data.TextLineDataset(FLAGS.train_file_path)
        dataset = dataset.skip(1)

        FLAGS.steps_per_epoch = counts // FLAGS.batch_size + \
            (counts % FLAGS.batch_size > 0) * 1
        print("train file has {} lines\ train_data will been split into {} batches(each batch size is {})".format(
            counts, FLAGS.steps_per_epoch, FLAGS.batch_size))

        # 1.2  划分 train/dev/test set
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(
            buffer_size=FLAGS.buffer_size))

        train_counts = int(0.7 * counts)
        val_counts = int(0.15 * counts)
        test_counts = int(0.15 * counts)
        test_dataset = dataset.take(test_counts)
        train_dev_dataset = dataset.skip(test_counts)

        def get_train_val_set(dataset):
            def split_train_val_set(dataset, train_dataset_index_list, val_dataset_index_list):
                dataset_split_dict = {}
                for fold_id in range(FLAGS.folds):
                    dataset_split_dict[fold_id] = dataset.shard(
                        FLAGS.folds, fold_id)

                val_dataset_index_list = [0, 2]
                train_dataset_index_list = [1, 3]

                def concatenate_datasets(dataset_split_dict, dataset_index_list):
                    dataset = dataset_split_dict[dataset_index_list[0]]

                    num_of_dataset_index = len(dataset_index_list)
                    if num_of_dataset_index > 2:
                        for i in range(1, num_of_dataset_index):
                            dataset = dataset.concatenate(
                                dataset_split_dict[dataset_index_list[i]])
                    return dataset
                val_dataset = concatenate_datasets(
                    dataset_split_dict, val_dataset_index_list)
                train_dataset = concatenate_datasets(
                    dataset_split_dict, train_dataset_index_list)
                return train_dataset, val_dataset

            k_fold = KFold(n_splits=FLAGS.folds)
            n = 0
            for train_dataset_index_list, val_dataset_index_list in k_fold.split(list(range(FLAGS.folds))):
                if n == 0:
                    train_dataset, val_dataset = split_train_val_set(
                        dataset, train_dataset_index_list, val_dataset_index_list)
                else:
                    train_dataset_2, val_dataset_2 = split_train_val_set(
                        dataset, train_dataset_index_list, val_dataset_index_list)
                    train_dataset = train_dataset.concatenate(train_dataset_2)
                    val_dataset = val_dataset.concatenate(val_dataset_2)
                n += 1

            return train_dataset, val_dataset
        train_dataset, val_dataset = get_train_val_set(train_dev_dataset)

        # 2 转换
        def dataset_map_batch(dataset):
            def decode_train_line(line):
                # Decode the csv line to tensor
                record_defaults = [[1.0] for col in range(785)]
                items = tf.decode_csv(line, record_defaults)
                features = items[1:785]
                features = tf.cast(features, tf.float32)
                features = tf.reshape(features, FLAGS.input_shape)

                label = items[0]
                label = tf.cast(label, tf.int32)
                label = tf.reshape(label, [1])
                # label = tf.one_hot(label, 10)
                return features, label

            dataset = dataset.apply(tf.data.experimental.map_and_batch(
                map_func=decode_train_line, batch_size=FLAGS.batch_size, num_parallel_calls=FLAGS.num_parallel_calls))
            dataset = dataset.prefetch(FLAGS.prefetch_buffer_size)
            return dataset

        train_dataset = dataset_map_batch(train_dataset)
        val_dataset = dataset_map_batch(val_dataset)
        test_dataset = dataset_map_batch(test_dataset)

        train_dataset = train_dataset.repeat(FLAGS.epochs)
        val_dataset = val_dataset.repeat(FLAGS.epochs)

        # ---------****[predict_data]****---------
        predict_x_dataset = tf.data.TextLineDataset(FLAGS.predict_x_path)
        predict_x_dataset = predict_x_dataset.skip(1)

        with open(FLAGS.predict_x_path, "r") as f:
            for counts, __ in enumerate(f):
                pass
        FLAGS.predict_x_steps_per_epoch = counts // FLAGS.predict_x_batch_size + \
            (counts % FLAGS.predict_x_batch_size > 0) * 1
        print("predict_x file has {} lines\ predict_x_data will been split into {} batches(each batch size is {})".format(
            counts, FLAGS.predict_x_steps_per_epoch, FLAGS.predict_x_batch_size))

        # 2. T
        def decode_predict_line(line):
            # Decode the csv line to tensor
            record_defaults = [[1.0] for col in range(784)]
            items = tf.decode_csv(line, record_defaults)
            features = items
            features = tf.cast(features, tf.float32)
            features = tf.reshape(features, FLAGS.input_shape)
            return features

        predict_x_dataset = predict_x_dataset.apply(tf.data.experimental.map_and_batch(
            map_func=decode_predict_line, batch_size=FLAGS.predict_x_batch_size, num_parallel_calls=FLAGS.num_parallel_calls))

        predict_x_dataset = predict_x_dataset.prefetch(1)

        # 3. Load
        # ---------****[train_data]****---------
        train_iterator = train_dataset.make_one_shot_iterator()
        #train_iterator = train_dataset.make
        train_x, train_y = train_iterator.get_next()

        val_iterator = val_dataset.make_one_shot_iterator()
        val_x, val_y = val_iterator.get_next()

        # ---------****[predict_data]****---------
        predict_x_iterator = predict_x_dataset.make_one_shot_iterator()
        predict_x = predict_x_iterator.get_next()

        """
        如果使用feed_dict
        x, y_ 使用tf. placehold 处理
        """
        global device_id
        device_id = -1

        def next_device(use_cpu=True):
            ''' See if there is available next device;
                Args: use_cpu, global device_id
                Return: new device id
            '''
            global device_id
            if use_cpu:
                if (device_id + 1) < FLAGS.num_cpu_core:
                    device_id += 1
                device = '/cpu:%d' % device_id
            else:
                if ((device_id + 1) < FLAGS.num_gpu_core):
                    device_id += 1
                device = '/gpu:%d' % device_id
            return device

        def variable_summaries(var):
            """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
            with tf.name_scope('summaries'):
                # 计算参数的均值，并使用tf.summary.scaler记录
                mean = tf.reduce_mean(var)
                tf.summary.scalar(var.name+'-mean', mean)

            # 计算参数的标准差
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            # 使用tf.summary.scaler记录记录下标准差，最大值，最小值
            tf.summary.scalar(var.name+'-stddev', stddev)
            tf.summary.scalar(var.name+'-max', tf.reduce_max(var))
            tf.summary.scalar(var.name+'-min', tf.reduce_min(var))
            # 用直方图记录参数的分布
            tf.summary.histogram(var.name+'-histogram', var)

        def build_model(x):
            # with tf.device(next_device()):
            # --------[PART 02] build model --------------
            # ----[1]  Forward

            conv1 = tf.layers.conv2d(x, filters=6,
                                     kernel_size=[3, 3], padding="same",
                                     activation=tf.nn.tanh,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.initializers.zeros(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

            pool1 = tf.layers.max_pooling2d(
                inputs=conv1, pool_size=[2, 2], strides=2)

            # with tf.device(next_device()):

            conv2 = tf.layers.conv2d(pool1, filters=16,
                                     kernel_size=[5, 5], padding="same", activation=tf.nn.tanh, kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.initializers.zeros(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))

            pool2 = tf.layers.max_pooling2d(
                inputs=conv2, pool_size=[2, 2], strides=2)

            pool2_flat = tf.layers.flatten(pool2)
            # with tf.device(next_device()):
            """

            dense = tf.layers.dense(
                inputs=pool2_flat, units=1024, activation=tf.nn.tanh)
            """

            dense = tf.layers.dense(
                inputs=pool2_flat, units=120, activation=tf.nn.tanh)

            logits = tf.layers.dense(inputs=dense, units=10)
            return logits

        if True:
            """
            """
            # 预测值
            train_predict_logits = build_model(train_x)
            train_predict_label = tf.argmax(
                tf.nn.softmax(train_predict_logits), axis=1)
            # train_predict_label = tf.reshape(train_predict_label, [-1, 1])
            # 评价值
            train_accuracy = tf.metrics.accuracy(
                labels=train_y, predictions=train_predict_label)
            # 验证值
            val_predict_labels = tf.argmax(
                input=tf.nn.softmax(build_model(val_x)), axis=1)
            val_accuracy = tf.metrics.accuracy(
                labels=val_y, predictions=val_predict_labels)

            labels = tf.argmax(input=build_model(predict_x), axis=1)

        # ----[2].BackProp (loss ,opt )
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=train_y, logits=train_predict_logits, name="haha")
            l2_loss = tf.losses.get_regularization_loss()

            object_function = loss+l2_loss

        # ----[3] Tensorboard---
            with tf.device(next_device()):
                variable_summaries(loss)
                variable_summaries(val_accuracy[1])
                for var in tf.trainable_variables():
                    tf.summary.histogram(var.name, var)

        # --------[PART 03 ] define  OP  --------------
        # with tf.device(next_device()):
            global_steps = tf.train.get_or_create_global_step()
            train_op = tf.train.AdamOptimizer(
                FLAGS.learning_rate).minimize(object_function, global_step=global_steps)

    # ------------[PART 2] Session Run --------------
    # 1. Define Session
    # logdir用来保存 checkpoint和 summary

        config = tf.ConfigProto(allow_soft_placement=True,
                                intra_op_parallelism_threads=0,
                                inter_op_parallelism_threads=0)
        # 1. Train
        with tf.train.MonitoredTrainingSession(config=config,
                                               checkpoint_dir=FLAGS.model_save_path,
                                               save_summaries_steps=10,
                                               summary_dir=FLAGS.log_dir_path) as sess:
            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            epoch = 0
            # chrome_trace

            for step in range(FLAGS.epochs*FLAGS.steps_per_epoch*FLAGS.folds):
                if step < 10:
                    sess.run([train_op], options=options,
                             run_metadata=run_metadata)
                    fetched_timeline = timeline.Timeline(
                        run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(os.path.join(FLAGS.profile_dir_path, str(datetime.datetime.now().minute) + 'timeline-%s.json' % step), 'w') as f:
                        f.write(chrome_trace)
                else:
                    sess.run([train_op])
                if step % FLAGS.steps_per_epoch == 0:
                    accuracy = sess.run(train_accuracy)
                    epoch += 1
                    print("{}/{} accuracy is {} ".format(epoch,
                                                         FLAGS.epochs*FLAGS.folds, accuracy))
                    if epoch % 5 == 0:
                        val_acc = sess.run(val_accuracy)
                        print("epoch ={}, val_accuracy is{}".format(epoch, val_acc))

        # 2. predict
        if PREDICT_ON:
            with tf.Session(config=config) as sess:
                saver = tf.train.import_meta_graph(
                    FLAGS.model_save_path+"/model.ckpt-0.meta")
                saver.restore(sess, tf.train.latest_checkpoint(
                    FLAGS.model_save_path))

                df_predict_label = pd.DataFrame()
                predict_label = np.array([])
                for i in range(FLAGS.predict_x_steps_per_epoch):
                    predict_label_i = sess.run(labels)
                    predict_label = np.concatenate(
                        [predict_label, predict_label_i], axis=0)
                df_predict_label = pd.DataFrame(
                    {"ImageId": range(1, len(predict_label)+1), "Label": predict_label})
                df_predict_label.to_csv(
                    FLAGS.predict_label_output_path, index=False)
        print("Finished")


def process_1():
    tf.app.run()
    print("Finished")


def process_2():
    print("This is process_2")

    os.system('tensorboard --logdir='+FLAGS.log_dir_path)


def process_3():
    print("This is process_3")


if __name__ == "__main__":
    t1 = threading.Thread(target=process_1, name="process_1")
    t1.start()
    t2 = threading.Thread(target=process_2, name="process_2")
    t2.start()
    t3 = threading.Thread(target=process_3, name="process_3")
    t3.start()
