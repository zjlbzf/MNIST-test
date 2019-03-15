

[toc]

# Tensorflow 的几个模块理解
## 1.训练的监视
tensorflow 主要有2种方式进行训练监督
1. tf.train.MonitorSession()
2. tf.train.Supervisor


### 1. tf.train.Supervisor

tf.Supervisor 是一个比较简单的训练监督方法。

> 1. 自动save_model
> 2. 自动write_summary (tensorboard)
> 3. 检查停止点
> 4. check_point 文件中恢复模型
> 

* 它每10分钟(默认save_model_secs=600)向 'logdir'内保存图内的 vars
* 并且它每2分钟（默认save_summaries_secs=120）自动运行 所有的 summary_ops，同时把 event file 存进 'logdir' 。
* 他还自动记录 steps ， 在它自己的线程里启动  tf.train.QueueRunner 。
* 也可以用它来检查停止点 tf.Supervisor.should_stop() ，要使用这个的话需要自己来设定 stop creterion ，然后当满足该条件时再用 tf.Supervisor.request_stop() 来触发 tf.Supervisor.should_stop() ，下一次检查 should_stop() 时就会停下。


> 
        sv.prepare_or_wait_for_session(server.target)
        merged_summary_op = tf.merge_all_summaries()
        sv = tf.train.Supervisor(logdir="/home/keith/tmp/",init_op=init_op) 
        #logdir用来保存checkpoint和summary
        saver=sv.saver

从上面代码可以看出，Supervisor帮助我们处理一些事情：
1. 自动去checkpoint加载数据或初始化数据
2. 自身有一个Saver，可以用来保存checkpoint
3. 有一个summary_computed用来保存Summary

所以，我们就不需要：
1. 手动初始化或从checkpoint中加载数据
2. 不需要创建Saver，使用sv内部的就可以
3. 不需要创建summary writer


        __init__(
        graph=None,
        ready_op=USE_DEFAULT,
        ready_for_local_init_op=USE_DEFAULT,
        is_chief=True,
        init_op=USE_DEFAULT,
        init_feed_dict=None,
        local_init_op=USE_DEFAULT,
        logdir=None,
        summary_op=USE_DEFAULT,
        saver=USE_DEFAULT,
        global_step=USE_DEFAULT,
        save_summaries_secs=120,
        save_model_secs=600,
        recovery_wait_secs=30,
        stop_grace_secs=120,
        checkpoint_basename='model.ckpt',
        session_manager=None,
        summary_writer=USE_DEFAULT,
        init_fn=None,
        local_init_run_options=None
        )



### 2. tf.train.MonitorSession()

处理以下操作：
1. 参数初始化
2. 使用钩子hook
3. 从错误中恢复会话的运行

执行步骤:
> 1. MonitoredSession init
> 2. MonitoredSession run
> 3. MonitoredSession close

#### 1. MonitoredSession init 初始化
>
1. 调用 hook.begin()
2. 调用scaffold.finalize() 
   >初始化计算图
   >> 如果没有指定scaffold,使用 ops.GraphKeys.SUMMARY_OP 构建图，所有， tf.train.MonitorSession() 要指定图。tf.1.12 的init 中
3. Creatsession.
   >sess=tf.Seession as
   创建会话 
4. init Scaffold的操作(op)
   > sess.run(Scaffold.init_op) 初始化模型
   >>Scaffold.init_op 如果没有，

5. 如果checkpoint存在的话，restore模型的参数 launches queue runners
6. 调用hook.after_create_session()

#### 2. MonitoredSession run 运行
>
1. 调用hook.before_run()
2. 调用TensorFlow的 session.run()
3. 调用hook.after_run()
4. 返回用户需要的session.run()的结果

如果发生了AbortedError或者UnavailableError，则在再次执行run()之前恢复或者重新初始化会话
#### 3.  MonitoredSession close 关闭
>
1. 调用hook.end()
2. 关闭队列和会话
3. 阻止OutOfRange错误
>

#### 4. 关键点1： global step

global_step在训练中是计数的作用，每训练一个batch就加1

在初始化训练的监控程序之前，一个用于跟踪训练步数的张量 global step 必须 添加到图中

1. 初始化时，global step必须被设定:
   
        global_step = tf.train.get_global_step /tf.train.get_or_create_global_step
2. 在训练过程中，step可以被获取:
   
        step = tf.train.global_step(sess, global_step)
3. step 通过传递到min op 中自增加:
   
        tf.train.AdamOptimizer(self.learning rt).minimize(self.loss, global step=self.step)
*****

>**Arg** :
**session_creator**: 制定用于创建回话的ChiefSessionCreator.
**hooks**: An iterable of `SessionRunHook' objects.


> 主要有
> 1.  LoggingTensorHook  每 N step/每N 秒 输出指定 tensors
> 2.  SummarySaverHook 每N steps 保存Summary
> 3.  CheckpointSaverHook 每N steps 保存Checkpoint
> 4. NanTensorHook 监测loss 函数，当loss==NaN 时停止训练
> 5. FeedFnHook 运行 feed_fn 函数 通知是的 feed_dict 字典匹配 
> 6. GlobalStepWaiterHook 分布式使用GlobalStep时 
> 7. ProfilerHook 在分布式计算过程中用于逐步启动worker
> 8. tf.train.StopAtStepHook()设置停止训练的条件

#### 5. 关键点2： tf.train.Scaffold

当你建立以用于训练的模型的时候，你通常需要 初始化操作、一个保存检查点checkpoint 的操作，一个用于tensorboard 可视化的summary 操作等，Scaffold类 将帮助你创建并添加到一个集合里面

可以省略 init_op \ tensorboard  tf.summary.merge_all()
或者 在tf.train.Scaffold中定义 init_op、summary_op、ready_op

__init__(
    init_op=None,
    init_feed_dict=None,
    init_fn=None,
    ready_op=None,
    ready_for_local_init_op=None,
    local_init_op=None,
    summary_op=None,
    saver=None,
    copy_from_scaffold=None
)


#### 6. 常用子类： tf.train.MonitoredTrainingSession()
属于.MonitorSession() 是其子类 ,返回.MonitorSession()
主要功能：
1. 自动保存检查点checkpoint
2. 自动运行保存summary（tensorboard）
3. 方便在多设备上运行tensorflow

        tf.train.MonitoredTrainingSession(
        master='',
        is_chief=True,
        checkpoint_dir=None,
        scaffold=None,
        hooks=None,
        chief_only_hooks=None,
        save_checkpoint_secs=USE_DEFAULT,
        save_summaries_steps=USE_DEFAULT(100),
        save_summaries_secs=USE_DEFAULT,
        config=None,
        stop_grace_period_secs=120,
        log_step_count_steps=100,
        max_wait_secs=7200,
        save_checkpoint_steps=USE_DEFAULT,
        summary_dir=None
        )

>**Args:**
**master** 字符串 the TensorFlow master to use.
**is_chief**：用于分布式系统中，用于判断该系统是否是chief，如果为True，它将负责初始化并恢复底层TensorFlow会话。如果为False，它将等待chief初始化或恢复TensorFlow会话。
**checkpoint_dir**：字符串。指定一个用于恢复变量的checkpoint文件路径。
**scaffold** ：(脚手架) 用于集合或建立op 。如果未指定，则会创建默认一个默认的scaffold。它用于完成图
**hooks**：一个SessionRunHook对象的列表。
**chief_only_hooks**：SessionRunHook对象列表。如果is_chief== True，则激活这些挂钩，否则忽略。
**save_checkpoint_secs**：用默认的checkpoint saver保存checkpoint的频率（以秒为单位）。如果save_checkpoint_secs设置为None，不保存checkpoint。
**save_summaries_steps**：使用默认summaries saver将摘要写入磁盘的频率（以全局步数表示）。如果save_summaries_steps和**save_summaries_secs** 都设置为None，则不使用默认的summaries saver保存summaries。默认为100
**save_summaries_secs**：使用默认summaries saver将摘要写入磁盘的频率（以秒为单位）。如果save_summaries_steps和save_summaries_secs都设置为None，则不使用默认的摘要保存。默认未启用。
**config**：用于配置会话的tf.ConfigProtoproto的实例。它是tf.Session的构造函数的config参数。
**stop_grace_period_secs**：调用close（）后线程停止的秒数。
**log_step_count_steps**：记录全局步/秒的全局步数的频率

>**Return :**
一个MonitoredSession（） 实例。



#### tensorflow hook架构

所有的hook都继承自SessionRunHook，定义在session_run_hook.py 文件里。其包含五个通用接口：

        def begin(self)

        def after_create_session(self, session, coord)

        def before_run(self, run_context)

        def after_run(self, run_context, run_values)

        def end(self, session)



## 2. 设备的管理
### 2.1 tf.device
        tf.device(device_name_or_function)

>device_name_or_function:
>>1. 设备名称字符串 
>> /job:<JOB_NAME>/task:<TASK_INDEX>/device:<DEVICE_TYPE>:<DEVICE_INDEX>
>>2. 返回设备字符串的函数

**其中：**

+ **<JOB_NAME>** 是一个字母数字字符串，并且不以数字开头。
+ **<DEVICE_TYPE>** 是一种注册设备类型（例如 GPU 或 CPU）。

+ **<TASK_INDEX>** 是一个非负整数，表示名为 
+  **<JOB_NAME>** 的作业中的任务的索引。请参阅 tf.train.ClusterSpec 了解作业和任务的说明。
+ **<DEVICE_INDEX>** 是一个非负整数，表示设备索引，例如用于区分同一进程中使用的不同 GPU 设备。

>>2. 设备函数

### 当设备为分布式设备时候
tf.train.replica_device_setter(
            cluster=cluster
    )


****

## 3. 数据输入的管理（tf.data）
两部分组成（使用的时候也分2步）：
1. 构建数据集
2. 构建迭代器
### 3.2 构建迭代器 
构建怎么样的迭代器？
> 本质上还是 分为4步
> 1. 构建 数据集 dataset
> 2. 构建 迭代器 interator 
> 3. 构建 next 操作 get_next
> > next 操作得出相应数据变量 切入到模型中
> 4. 构建 初始化 操作  init_op
> > 初始化操作可能在运行时候被 全局一次性初始化替代



| 迭代器类型      | 难度  |                                       应用场景 |
| --------------- | :---: | ---------------------------------------------: |
| one-shot        |   0   | 仅支持对整个数据集访问一遍，不需要显式的初始化 |
| initializable   |   1   |                                       支持参数 |
| reinitializable |   2   |                   支持多个相同数据结构的数据集 |
| feedable        |   3   |                                   支持调用机制 |





1. one-shot

        # 构建dataset 
        dataset = tf.data.Dataset.range(100)
        # 构建 迭代器（生成器）
        iterator = dataset.make_one_shot_iterator() 
        # 构建 next op操作
        next_element = iterator.get_next()
        # 构建 初始化 操作（one-shot 模式 无）

        # run
        for i in range(100):
            value = sess.run(next_element)
            assert i == value、

2. initializable （可初始化的）
        
        # 构建dataset （数据集元素数量待定）
        max_value = tf.placeholder(tf.int64, shape=[])
        dataset = tf.data.Dataset.range(max_value)
        # 构建 迭代器（生成器）
        iterator = dataset.make_initializable_iterator()
        # 构建 next op操作
        next_element = iterator.get_next()
        # 构建 初始化 操作
        initializer_op=iterator.initializer

        # 例子1 初始化一个含有10个元素的迭代器
        sess.run(initializer_op, feed_dict={max_value: 10})
        for i in range(10):
                value = sess.run(next_element)
                assert i == value

        # 例子2 初始化一个含有100个元素的迭代器
        sess.run(initializer_op, feed_dict={max_value: 100})
        for i in range(100):
                value = sess.run(next_element)
                assert i == value
3. reinitializable （可重复初始化的（一般针对多个具有相同数据结构的数据集））
        
        # 构建dataset 
        # 定义一个具有相同结构的 训练数据集 和 验证集
        training_dataset = tf.data.Dataset.range(100).map(
        lambda x: x + tf.random_uniform([], -10, 10, tf.int64))
        validation_dataset = tf.data.Dataset.range(50)


        # 构建 迭代器（生成器）
        iterator = tf.Iterator.from_structure(training_dataset.output_types,training_dataset.output_shapes)
        
        # 构建 next op操作
        next_element = iterator.get_next()

        # 构建 初始化 操作
        training_init_op = iterator.make_initializer(training_dataset)
        validation_init_op = iterator.make_initializer(validation_dataset)

        for _ in range(20):
                # 初始化训练集的 迭代器.
                sess.run(training_init_op)
                for _ in range(100):
                        sess.run(next_element)

                # 初始化验证集的 迭代器.
                sess.run(validation_init_op)
                for _ in range(50):
                        sess.run(next_element)
4. feedable （可反馈的--正对有调用机制的）
  
   一个调用机制

> 1. 构建数据集
> 2. 针对数据集构建迭代器
> 3. 构建 next 操作 （使用handle）
>>   + 数据集建立 handle
>>   + 定义一种统一的迭代器类型（带有handle str 和数据结构）
>>   + 以统一的迭代器进行 next 操作

   定义完 操作后，可以通过feed 选择  

        # 构建不同的 数据集
        training_dataset = tf.data.Dataset.range(100).map(lambda x: x + tf.random_uniform([], -10, 10, tf.int64)).repeat()
        validation_dataset = tf.data.Dataset.range(50)

        
        # 为不同数据集构建 迭代器
        training_iterator = training_dataset.make_one_shot_iterator()
        validation_iterator = validation_dataset.make_initializable_iterator()

        # 为不同数据集构建 handle
        training_handle = sess.run(training_iterator.string_handle())
        validation_handle = sess.run(validation_iterator.string_handle())


        # 定义一个handle str 
        handle = tf.placeholder(tf.string, shape=[])
        
        ##定义一种数据结构的 iterator
        iterator = tf.data.Iterator.from_string_handle(
        handle, training_dataset.output_types, training_dataset.output_shapes)
        
        # 构建 next 操作
        next_element = iterator.get_next()

        
        while True:
                for _ in range(200):
                        sess.run(next_element, feed_dict={handle: training_handle})

                sess.run(validation_iterator.initializer)
                for _ in range(50):
                        sess.run(next_element, feed_dict={handle: validation_handle})

## 4. I/O 问题
tensorflow gfile文件操作详解

翻译过来就是（我的翻译水平还有待提高，哈哈，暂且看看吧）：

1、提供了一种类似于python文件 I/O操作的API；
2、提供了一种操作tensorflow C++文件系统的API；

https://zhuanlan.zhihu.com/p/31536538

## 5.持久化（保存和恢复）

https://www.tensorflow.org/guide/saved_model
### 1. 几个概念
>>1. **数据序列化**就是将对象或者数据结构转化成特定的格式，使其可在网络中传输，或者可存储在内存或者文件中。
廖雪峰：变量从内存中变成可存储或传输的过程称之为序列化
对象序列化后的数据格式可以是二进制，可以是XML，也可以是JSON等任何格式。
>>2. **反序列化则**是相反的操作，将对象从序列化数据中还原出来。
廖雪峰：把变量内容从序列化的对象重新读到内存里称之为反序列化
>>3. **protobuf** 
 Google Protocol Buffers 简称 Protobuf，类似json的一种数据格式，但不同的是他是二进制格式，性能好、效率高
文件以.proto结尾 
https://tensorflow.juejin.im/extend/tool_developers/index.html
ProtoBuf 实际上支持两种不同的文件保存格式。
1.TextFormat 是一种人眼可读的文本形式，这在调试和编辑时是很方便的，但它在存储数值数据时会变得很大，比如我们常见的权重数据。文件名为 xxx.pbtxt。
2.二进制格式的文件会小得多，缺点就是它不易读。文件名为 xxx.pb
>>4. 
### 2. 需要保存什么
主要是：
#### 1. 图信息

|     名称 |                                                        解释 |                                                                 组成 |
| -------: | ----------------------------------------------------------: | -------------------------------------------------------------------: |
|    Graph | 被定义为“一些 Operation（节点） 和 Tensor（边缘） 的集合” |                                            Node(op+placeholder)+Edge |
| GraphDef |                                               序列化的Graph | NodeDef 的 Protocol Buffer，NodeDef对应 op的Node 和placeholder的node |

#### 2. 参数信息

其他：

1. 服务器信息
2. 集群信息

MetaGraphDef：
|          组成 |                                                                           内容 |                                                                              例如 |
| ------------: | -----------------------------------------------------------------------------: | --------------------------------------------------------------------------------: |
|   MetaInfoDef |                                                                   存一些元信息 |                                                                版本和其他用户信息 |
|      GraphDef |                                     Graph的序列化信息，MetaGraph的核心内容之一 |                                                            Node(Placeholder + op) |
|      SaverDef |                                                                  图的Saver信息 | 最多同时保存的check-point数量；需保存的Tensor名字等，但并不保存Tensor中的实际内容 |
| CollectionDef | 任何需要特殊注意的 Python 对象，需要特殊的标注以方便import_meta_graph 后取回。 |                                                     “train_op”,"prediction"等等 |



保存文件如下表所示：
### 1.ckpt类型（checkpoint）

.ckpt格式文件只能在tensorflow 框架下使用
其主要操作为saver

        saver = tf.train.Saver()

saver是一个tensorflow.python.training.saver.Saver 类 

saver的建立主要通过 tf.train 子类建立，有以下方式:
>1. saver = tf.train.Saver()
>2. saver = tf.train.import_meta_graph()
> 返回MetaGraphDef里面的saver_def 或者None

参数
        __init__(
        var_list=None,
        reshape=False,
        sharded=False,
        max_to_keep=5,
        keep_checkpoint_every_n_hours=10000.0,
        name=None,
        restore_sequentially=False,
        saver_def=None,
        builder=None,
        defer_build=False,
        allow_empty=False,
        write_version=tf.train.SaverDef.V2,
        pad_step_number=False,
        save_relative_paths=False,
        filename=None
        )
方法
>
1. as_saver_def
         返回一个 SaverDef proto
2. build 
3. export_meta_graph  返回一个MetaGraphDef
4. from_proto   从  saver_def 返回以一个Saver built
5. recover_last_checkpoints 从错误中恢复 saver的初始状态
6. restore  --恢复先前保存的参数变量
7. save -- 保存参数变量
8. set_last_checkpoints --
9. set_last_checkpoints_with_time
10. to_proto。将  Saver 转化成 a SaverDef protocol buffer.


|                            文件名 |   文件类型 |                             描述 |                                            包含 |
| --------------------------------: | ---------: | -------------------------------: | ----------------------------------------------: |
|                        checkpoint |   文本文件 | 可直接记事本打开，记录检查点信息 | model_checkpoint_path;all_model_checkpoint_path |
|                 model.ckpt-0.meta | 二进制文件 |                           图结构 |
|               model.ckpt-20.index | 二进制文件 |                       数据 index |
| model.ckpt-20.data-00000-of-00002 | 二进制文件 |                             数据 |



> **文件的命名** 
**model.ckpt-20**
mode：可变的文件名
ckpt：文件格式
-20： 代表 global_step =20 
saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
...
saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'

#### 1.保存
1. 创建saver
   
        saver=tf.train.Saver()
2. 保存参数变量
   
        saver_path = saver.save(sess, save_path ,global_step=100)

#### 2.加载
1. 加载持久化图
   
        saver=tf.train.import_meta_graph(“save/model.ckpt.meta”)
        

2. 加载保存的参数
        saver.restore(sess,tf.train.latest_checkpoint()/save_path)

        sess 必须提前加载，同时参数没有初始化，因为restore 方法本身就是一个初始化的过程



### 2. pd 格式
PB 文件是表示 MetaGraph 的 protocol buffer格式的文件

#### 1.保存

1. 通过 get_default_graph().as_graph_def() 得到当前图的计算节点（Node）信息

2. 通过 graph_util.convert_variables_to_constants 将相关节点的values固定值
   
        var_list = tf.trainable_variables()
        constant_graph = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, [var_list[i].name[:-2] for i in range(len(var_list))])
3. 通过 tf.gfile.GFile 进行模型持久化.保存trainable variables到.pb文件：

        with tf.gfile.FastGFile(pb_path, mode='wb') as f:
                f.write(constant_graph.SerializeToString())

#### 2.2 pd 文件加载

1. tf.io.gfile.GFile 打开pb文件
   
        model_f = tf.io.gfile.GFile("./Test/model.pb", mode='rb')
        # 无线程锁定的 I/O 处理
        tensorflow ->i/o->

2. 将pb文件输入到graph_def类中

2.1 创建一个空的graph_def 类

        graph_def = tf.GraphDef()

2.2 将 MetaGraph的二进制文件调入 空的graph_def，加载节点(Node)信息

        graph_def.ParseFromString(model_f.read())  ##二进制调用实际上是 ParseFromString

3. 将graph_def调入到现在的graph 中
   
        c = tf.graph_util.import_graph_def(graph_def, return_elements=["add2:0"])
        通过return_elements 确定返回的 op /tensor


4. 获得op/tensor ，可以运行
   
        print(sess.run(c))



## 6. 可视化（tf.summary）

tf.summary
Tensorflow 的总结操作-- （获取tensor，返回支持Tensorboard的tensor）
tf.summary和其他op 一样是属于操作范围的，输入tensor，返回 tensor

### 1 得到数据 （输入tensor,得到tensor的操作 ）

1. tf.summary.scalar(记录  标量)
2. tf.summary.image (记录 图像)
3. tf.summary.audio  (记录 音频)
4. tf.summary.text  (记录 文本)

****
1. tf.summary.histogram（记录 数据的直方图）
2. tf.summary.distribution (记录 数据的分布图)

**** 
> 上面的每一个op 都是 构建图的一部分，没有会话的执行sess.run 都不会计算

为了会话计算方便，可以把上面所有 op 合并为一个

        tf.summary.merge_all()

### 2  将输出的数据都保存到本地磁盘中  这是一个命令 不需要 sess run 

        tf.summary.FileWriter



## 7. 评价模型的优劣（tf.metrics）



tf.metrics.accuracy()
tf.metrics.precision()
tf.metrics.recall()
tf.metrics.mean_iou()
###1. 如何评价一个模型 

#### 1 误差 error

1. 绝对误差
mean_absolute_error(...): Computes the mean absolute error between the labels and predictions.

2. 相对误差
mean_relative_error(...): Computes the mean relative error by normalizing with the given values.

3. 平方差（方差）--离散度
mean_squared_error(...): Computes the mean squared error between the labels and predictions.

4. 均方差(标准差)--离散度
   root_mean_squared_error(...): Computes the root mean squared error between the labels and predictions.
5. 余弦相似性
mean_cosine_distance(...): Computes the cosine distance between the labels and predictions.


#### 2 混合矩阵


![avatar](https://upload-images.jianshu.io/upload_images/7252179-dbad746fab87dc42.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/646/format/webp)

1. 准确率 (accuracy)
   $$accuracy=正确预测的数量/样本总数$$
   
2. 精确率 (precision)
   模型正确预测正类别的频率
   >正类别：在二元分类中，两种可能的类别分别被标记为正类别和负类别。正类别结果是我们要测试的对象。例如，在医学检查中，正类别可以是“肿瘤”

   $$precision=正例数/(正例数+假正例数)$$

3. 召回率 (recall)
   在所有可能的正类别标签中，模型正确地识别出了多少个？即
   $$recall=正例数/(正例数+假负例数)$$
4. auc ROC 曲线下面积 (AUC, Area under the ROC Curve)
   >受试者工作特征曲线（receiver operating characteristic，简称 ROC 曲线）

auc(...): Computes the approximate AUC via a Riemann sum.


1. 敏感度(sensitivity)
   
2. 特异度(specificity)
sensitivity_at_specificity(...): Computes the specificity at a given sensitivity.


7.数量
false_negatives(...): Computes the total number of false negatives.
false_positives(...): Sum the weights of false positives.
true_negatives(...): Sum the weights of true_negatives.
true_positives(...): Sum the weights of true_positives.


#### 3 基本统计量 


mean_iou(...): Calculate per-step mean Intersection-Over-Union (mIOU).


1. 平均数
mean(...): Computes the (weighted) mean of the given values.

2. 百分百
percentage_below(...): Computes the percentage of values less than the given threshold.


## 8. 代码的可读性

为了解决这个问题，我们引入了 name_scope 和 variable_scope， 二者又分别承担着不同的责任：

* name_scope: * 为了更好地管理变量的命名空间而提出的。比如在 tensorboard 中，因为引入了 name_scope， 我们的 Graph 看起来才井然有序。
* variable_scope: * 大大大部分情况下，跟 tf.get_variable() 配合使用，实现变量共享的功能。

这两个函数在大部分情况下是等价的, 唯一的区别是在使用tf.get_variable函数时. 

        import tensorflow as tf

        with tf.variable_scope("foo"):
                a = tf.get_variable("bar", [1])
                print a.name    # 输出 foo/bar: 0

        with tf.variable_scope("bar"):
                b = tf.get_variable("bar", [1])
                print b.name     # 输出 bar/bar: 0

        with tf.name_scope("a"):
                a = tf.Variable([1])
                print a.name     # 输出 a/Variable: 0

        　　    a = tf.Variable("b", [1]):
        　　    print a.name　# 输出 b: 0

        with tf.name_scope("b"):
                tf.get_variable("b", [1])        # Error


## 9. Losses

损失函数的问题最终还是要归结于 任务类型，是处理predict_y 和 real_y 的问题

单个样本：损失函数（Loss Function）
多个样本：成本函数（Cost Function）

目标：在有约束条件下的最小化成本函数（Cost Function）


1. 离散数据
   1. 分类
      1. 二分类
      2. 多分类
2. 连续数据
   1. 回归

https://www.tensorflow.org/api_docs/python/tf/losses
https://heartbeat.fritz.ai/5-regression-loss-functions-all-machine-learners-should-know-4fb140e9d4b0

![avatar](https://cdn-images-1.medium.com/max/1600/1*3MsFzl7zRZE3TihIC9JmaQ.png)
![avatar](https://cdn-images-1.medium.com/max/1600/1*BploIBOUrhbgdoB1BK_sOg.png)


### 1. 神经网络里面的loss （tf.nn.loss）

#### 1. tf.nn.sigmoid_cross_entropy_with_logits
        
        (labels=None,logits=None,name=None)

        labels是真实y shape=[batchs,num_classe]，one-hot模式
        logits
        过程：
        1.y_=sigmoid(logits)
        2. 交叉熵

#### 2. tf.nn.softmax_cross_entropy_with_logits

        labels是真实y shape=[batchs,num_classe],one-hot 模式
        logits
        过程：
        1.y_=sigmoid(logits)
        2. 交叉熵

#### 3. tf.nn.sparse_softmax_cross_entropy_with_logits
        labels是真实y shape=[batchs,1 ]
        logits
        过程：
        1.y_=sigmoid(logits)
        1. 交叉熵


交叉熵=log loss
absolute_difference(...): Adds an Absolute Difference loss to the training procedure.

add_loss(...): Adds a externally defined loss to the collection of losses.

compute_weighted_loss(...): Computes the weighted loss.

cosine_distance(...): Adds a cosine-distance loss to the training procedure. (deprecated arguments)

get_losses(...): Gets the list of losses from the loss_collection.

get_regularization_loss(...): Gets the total regularization loss.

get_regularization_losses(...): Gets the list of regularization losses.

get_total_loss(...): Returns a tensor whose value represents the total loss.

hinge_loss(...): Adds a hinge loss to the training procedure.

huber_loss(...): Adds a Huber Loss term to the training procedure.

log_loss(...): Adds a Log Loss term to the training procedure.

mean_pairwise_squared_error(...): Adds a pairwise-errors-squared loss to the training procedure.

mean_squared_error(...): Adds a Sum-of-Squares loss to the training procedure.

sigmoid_cross_entropy(...): Creates a cross-entropy loss using tf.nn.sigmoid_cross_entropy_with_logits.

softmax_cross_entropy(...): Creates a cross-entropy loss using tf.nn.softmax_cross_entropy_with_logits_v2.

sparse_softmax_cross_entropy(...): Cross-entropy loss using tf.nn.sparse_softmax_cross_entropy_with_logits.

# 基于Tensorflow 的模型训练 基本步骤

## 1. 数据输入 （ETL）
###  1.1 Extrat提取（来源：文件、数据库）
###  1.2 Trans 转换（最后repeat）
###  1.3 Load 加载 

## 2. 图的绘制
 graph定义了computation，它不计算任何东西，不包含任何值，只是定义了你在代码中指定的操作。其中图中最重要的工作就是定义op

### 2.1 定义 op
>  先定义 train op
>  最后定义init op

## 3. 会话的运行
> 执行op