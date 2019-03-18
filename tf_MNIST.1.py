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
flags.DEFINE_integer("batch_size", 128, "")
flags.DEFINE_integer("num_parallel_calls", 4, "")
flags.DEFINE_float("learning_rate", 0.00005, "")
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
model_save_path = os.path.join(current_dir_path, "model", par_str)
predict_label_output_path = os.path.join(
    current_dir_path, "submit", par_str+".csv")
LOG_DIR_PATH = os.path.join(
    current_dir_path, "log", "train", par_str)
# predict_label_output_path=os.path.join(current_dir_path, "submit", par_str".h5")

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


def main(argv):

    FLAGS.learning_rate = argv[0]
    par_str = "lr_%g_b1_%g_b2_%g_bsize_%g-%s" % (
        FLAGS.learning_rate, FLAGS.beta_1, FLAGS.beta_2, FLAGS.batch_size, str(uuid.uuid1()))
    print("par_str is "+par_str)
    model_save_path = os.path.join(current_dir_path, "model", par_str)
    predict_label_output_path = os.path.join(
        current_dir_path, "submit", par_str+".csv")
    LOG_DIR_PATH = os.path.join(
        current_dir_path, "log", "train", par_str)
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

        if True:
            # with tf.device(next_device()):

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
                        train_dataset = train_dataset.concatenate(
                            train_dataset_2)
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
                    label = tf.one_hot(label, 10)
                    #label = tf.reshape(label, [10])
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
            val_iterator = val_dataset.make_one_shot_iterator()
            train_x, train_y = train_iterator.get_next()
            val_x, val_y = val_iterator.get_next()

            # ---------****[predict_data]****---------
            predict_x_iterator = predict_x_dataset.make_one_shot_iterator()
            predict_x = predict_x_iterator.get_next()

            def variable_summaries(var):
                """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
                with tf.name_scope(var.name.split(":")[0]):
                    # 计算参数的均值，并使用tf.summary.scaler记录
                    mean = tf.reduce_mean(var)
                    tf.summary.scalar('mean', mean)
                    tf.summary.scalar('stddev', tf.sqrt(
                        tf.reduce_mean(tf.square(var - mean))))
                    tf.summary.scalar('-max', tf.reduce_max(var))
                    tf.summary.scalar('min', tf.reduce_min(var))
                    # 用直方图记录参数的分布
                    tf.summary.histogram('histogram', var)

        # with tf.device(next_device()):
            # --------[PART 02] build model --------------
            # ----[1]  Forward
            is_training = tf.placeholder(tf.bool, shape=())
            x = tf.cond(is_training, lambda: train_x, lambda: val_x)
            y_ = tf.cond(is_training, lambda: train_y, lambda: val_y)

            conv1 = tf.layers.conv2d(x, filters=6,
                                     kernel_size=[3, 3], padding="same",
                                     activation=tf.nn.tanh,
                                     strides=1,
                                     kernel_initializer=tf.glorot_uniform_initializer(),
                                     bias_initializer=tf.initializers.zeros(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))

            pool1 = tf.layers.max_pooling2d(
                inputs=conv1, pool_size=[2, 2], strides=2, padding='valid')

            conv2 = tf.layers.conv2d(pool1, filters=16,
                                     kernel_size=[5, 5], padding="same", activation=tf.nn.tanh, kernel_initializer=tf.glorot_uniform_initializer(),
                                     strides=1,
                                     bias_initializer=tf.initializers.zeros(),
                                     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.0))

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

            logits = tf.layers.dense(
                inputs=dense, units=10, activation=tf.nn.softmax)

            predict_label = tf.argmax(logits, axis=1)
            print(y_)
            real_label = tf.argmax(y_, axis=1)
            print(real_label)

            accuracy_op = tf.metrics.accuracy(
                labels=real_label, predictions=predict_label)

            # ----[2].BackProp (loss ,opt )
            loss = tf.reduce_mean(-tf.reduce_sum(
                y_ * tf.log(logits), reduction_indices=[1]), name="cross_entropy")

            l2_loss = tf.losses.get_regularization_loss()
            object_function = loss + l2_loss
            global_step = tf.train.get_or_create_global_step()

        # ----[3] Tensorboard---
        # with tf.device(next_device()):
            variable_summaries(loss)
            variable_summaries(accuracy_op[1])
            for var in tf.trainable_variables():
                tf.summary.histogram(var.name, var)
            summary_op = tf.summary.merge_all()
        # --------[PART 03 ] define  OP  --------------
        # with tf.device(next_device()):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = tf.train.AdamOptimizer(
                    FLAGS.learning_rate).minimize(object_function, global_step=global_step)

        # ------------[PART 2] Session Run --------------
        # 1. Define Session
        # logdir用来保存 checkpoint和 summary

        config = tf.ConfigProto(
            device_count={"CPU": 8},
            log_device_placement=False,
            allow_soft_placement=False,
            intra_op_parallelism_threads=2,
            inter_op_parallelism_threads=2)
        summary_writer = tf.summary.FileWriter(LOG_DIR_PATH, graph)

        # 1. Train
        ChiefSessionCreator = tf.train.ChiefSessionCreator(
            config=config, checkpoint_dir=model_save_path)
        summary_hook = tf.train.SummarySaverHook(
            save_steps=20, output_dir=LOG_DIR_PATH, summary_writer=summary_writer, summary_op=summary_op)
        saver_hook = tf.train.CheckpointSaverHook(
            checkpoint_dir=model_save_path, save_steps=10)

        with tf.train.MonitoredSession(session_creator=ChiefSessionCreator, hooks=[summary_hook, saver_hook]) as sess:

            options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()

            epoch = 0
            # chrome_trace
            Total_steps = FLAGS.epochs * FLAGS.steps_per_epoch * FLAGS.folds
            for step in range(10):
                sess.run([train_op], feed_dict={is_training: True}, options=options,
                         run_metadata=run_metadata)

                fetched_timeline = timeline.Timeline(
                    run_metadata.step_stats)
                chrome_trace = fetched_timeline.generate_chrome_trace_format()
                with open(os.path.join(FLAGS.profile_dir_path, str(datetime.datetime.now().minute) + 'timeline-%s.json' % step), 'w') as f:
                    f.write(chrome_trace)

            for step in range(Total_steps-10):
                sess.run([train_op], feed_dict={is_training: True})

                if step % FLAGS.steps_per_epoch == 0:
                    accuracy = sess.run(accuracy_op, feed_dict={
                                        is_training: True})
                    epoch += 1
                    print("{}/{} accuracy is {} ".format(epoch,
                                                         FLAGS.epochs*FLAGS.folds, accuracy))
                    if epoch % 5 == 0:
                        val_acc = sess.run(accuracy_op, feed_dict={
                            is_training: False})
                        print("epoch ={}, val_accuracy is{}".format(epoch, val_acc))

        # 2. predict
        if PREDICT_ON:
            with tf.Session(config=config) as sess:
                saver = tf.train.import_meta_graph(
                    model_save_path+"/model.ckpt-0.meta")
                saver.restore(sess, tf.train.latest_checkpoint(
                    model_save_path))

                df_predict_label = pd.DataFrame()
                predict_label = np.array([])
                for i in range(FLAGS.predict_x_steps_per_epoch):
                    predict_label_i = sess.run(labels)
                    predict_label = np.concatenate(
                        [predict_label, predict_label_i], axis=0)
                df_predict_label = pd.DataFrame(
                    {"ImageId": range(1, len(predict_label)+1), "Label": predict_label})
                df_predict_label.to_csv(
                    predict_label_output_path, index=False)
        print("Finished")


def process_1():
    print("This is process 1！")
    tf.app.run(argv=[0.0001])


def process_2():
    print("This is process 2！")
    tf.app.run(argv=[0.0005])


if __name__ == "__main__":
    t1 = threading.Thread(target=process_1)
    t2 = threading.Thread(target=process_2)
    t1.start()
    t2.start()
