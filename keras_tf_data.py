#!/usr/bin/python3
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

try:
    current_path = os.path.split(os.path.realpath(__file__))[0]
except:
    current_path = os.path.split(os.path.realpath("__file__"))[0]

flags = tf.app.flags
# [1] System
flags.DEFINE_integer("buffer_size", 10, "buffer_size")
flags.DEFINE_string("current_path", current_path, "")

# [2] Dataset
flags.DEFINE_string("train_file_path", os.path.join(
    current_path, "raw_data", "train.csv"), "train_file_path")
flags.DEFINE_list("input_shape", [28, 28, 1], "input_shape")

# [3] Build model
flags.DEFINE_integer("hidden_units", 1, "hidden_units")
flags.DEFINE_string("build_model_type", "Build", "[Load Build]")
flags.DEFINE_string("load_model_path", os.path.join(
    current_path, "model", "default_model.h5"), "if load")

# [4] Train
flags.DEFINE_integer("batch_size", 128, "")
flags.DEFINE_integer("num_parallel_calls", 8, "")
flags.DEFINE_float("learning_rate", 0.0001, "")
flags.DEFINE_float("beta_1", 0.9, "")
flags.DEFINE_float("beta_2", 0.991, "")
flags.DEFINE_float("epsilon", 1e-8, "")
flags.DEFINE_bool("decay", False, "")
flags.DEFINE_integer("epochs", 100, "")
flags.DEFINE_integer("steps_per_epoch", 100, "default is 100")

# [5] Output
flags.DEFINE_string("record_type", "record_parameters", "")
flags.DEFINE_string("predict_x_path", os.path.join(
    current_path, "raw_data", "test.csv"), "")
flags.DEFINE_string("log_file", os.path.join(
    current_path, "log"), "")
flags.DEFINE_string("model_save_path", os.path.join(
    current_path, "model", "default_model.h5"), "")
flags.DEFINE_string("history_output_path", os.path.join(
    current_path, "train_process", "default_train_process.csv"), "")
flags.DEFINE_string("predict_label_output_path", os.path.join(
    current_path, "submit", "default_submit.csv"), "")

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


FLAGS = flags.FLAGS
IMAGE_PIXELS = 28

server = tf.train.Server.create_local_server()


def Timer(func):
    def newFunc(*args, **args2):
        t1 = datetime.datetime.now()
        back = func(*args, **args2)
        print(" This function【{}】cost time:{} \n".format(
            func.__name__, datetime.datetime.now()-t1))
        return back
    return newFunc


def main(__):

    print(f"Tensorflow Version is {tf.__version__}")

    # mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    ps_spec = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 创建集群
    num_worker = len(worker_spec)
    print("Cluster num is {}".format(num_worker))
    cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == 'ps':
        server.join()

    is_chief = (FLAGS.task_index == 0)
    worker_device = '/job:worker/task:%d/cpu:0' % FLAGS.task_index
    with tf.device(tf.train.replica_device_setter(cluster=cluster, worker_device=worker_device)):
        # with tf.device("/cpu:0"):
        # --------[PART 01] build model --------------
        # ----[0] init  (1 V 2 placehold)
        # 0.1 Variable
        # 0.2 placeholder
        global_step = tf.Variable(
            0, name='global_step', trainable=False)  # 创建纪录全局训练步数变量
        hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, FLAGS.hidden_units],
                                                stddev=1.0 / IMAGE_PIXELS), name='hid_w')
        hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name='hid_b')

        sm_w = tf.Variable(tf.truncated_normal([FLAGS.hidden_units, 10],
                                               stddev=1.0 / math.sqrt(FLAGS.hidden_units)), name='sm_w')
        sm_b = tf.Variable(tf.zeros([10]), name='sm_b')

        x = tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS])
        y_ = tf.placeholder(tf.float32, [None, 10])  # real_y

        # ----[1]  Forward

        hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
        hid = tf.nn.relu(hid_lin)
        y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

        #----[2].BackProp (loss ,opt )

        cross_entropy = - \
            tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
        opt = tf.train.AdamOptimizer(FLAGS.learning_rate)
        train_step = opt.minimize(cross_entropy, global_step=global_step)

        # 生成本地的参数初始化操作init_op
        init_op = tf.global_variables_initializer()
        #train_dir = tempfile.mkdtemp()
        sv = tf.train.Supervisor(is_chief=is_chief, logdir=FLAGS.log_file, init_op=init_op, recovery_wait_secs=1,
                                 global_step=global_step)

        if is_chief:
            print('Worker %d: Initailizing session...' % FLAGS.task_index)
        else:
            print('Worker %d: Waiting for session to be initaialized...' %
                  FLAGS.task_index)
        # --------[PART 02 ] Train  model --------------
        run_metadata = tf.RunMetadata()
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        config = tf.ConfigProto(graph_options=tf.GraphOptions(
            optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))

        sess = sv.prepare_or_wait_for_session(server.target)

        print('Worker %d: Session initialization  complete.' % FLAGS.task_index)
        time_begin = time.time()
        print('Traing begins @ %f' % time_begin)
        local_step = 0

        while True:
            batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch_size)
            train_feed = {x: batch_xs, y_: batch_ys}
            """
            feed_dict 输入数据是最慢的
            _, step = sess.run([train_step, global_step], feed_dict=train_feed)
            """

            local_step += 1

            now = time.time()
            print('%f: Worker %d: traing step %d dome (global step:%d)' %
                  (now, FLAGS.task_index, local_step, step))

            if step >= FLAGS.train_steps:
                break

        time_end = time.time()
        print('Training ends @ %f' % time_end)
        train_time = time_end - time_begin
        print('Training elapsed time:%f s' % train_time)

        val_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        val_xent = sess.run(cross_entropy, feed_dict=val_feed)
        print('After %d training step(s), validation cross entropy = %g' %
              (FLAGS.train_steps, val_xent))
        sess.close()

    # ------------[Part 4] Output   ----------------
    if FLAGS.record_type == "record_parameters":
        par_str = "lr_%g_b1_%g_b2_%g_bsize_%g" % (
            FLAGS.learning_rate, FLAGS.beta_1, FLAGS.beta_2, FLAGS.batch_size)
        FLAGS.model_save_path = os.path.join(
            FLAGS.current_path, "model", par_str + ".h5")
        FLAGS.history_output_path = os.path.join(
            FLAGS.current_path, "train_process", par_str + ".csv")
        FLAGS.predict_label_output_path = os.path.join(
            FLAGS.current_path, "submit", par_str+".csv")

    # 1 . model save
    model.save(FLAGS.model_save_path)
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(FLAGS.history_output_path)

    # 2. predict
    predict_data = pd.read_csv(FLAGS.predict_x_path)

    predict_x = predict_data.values.reshape([len(predict_data)] + [28, 28, 1])
    predict_y = model.predict(predict_x)
    # label
    predict_label = np.argmax(predict_y, axis=1)
    df_predict_label = pd.DataFrame(
        {"ImageId": range(1, len(predict_label)+1), "Label": predict_label})
    # export
    df_predict_label.to_csv(FLAGS.predict_label_output_path, index=False)


if __name__ == "__main__":

    tf.app.run()
