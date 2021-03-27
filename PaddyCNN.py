import tensorflow as tf
import numpy as np
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_and_decode(filename, batch_size):
    # 建立文件名隊列
    filename_queue = tf.train.string_input_producer([filename],
                                                    num_epochs=None)

    # 數據讀取器
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)

    # 數據解析
    img_features = tf.parse_single_example(
        serialized_example,
        features={'Label': tf.FixedLenFeature([], tf.int64),
                  'image_raw': tf.FixedLenFeature([], tf.string), })

    image = tf.decode_raw(img_features['image_raw'], tf.uint8)
    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.cast(image, tf.float32) * (1. / 255)
    label = tf.cast(img_features['Label'], tf.int64)

    # 依序批次輸出 / 隨機批次輸出
    # tf.train.batch / tf.train.shuffle_batch
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        num_threads=4,
        batch_size=batch_size,
        capacity=10000 + 3 * batch_size,
        min_after_dequeue=10000
    )

    # tf.train.shuffle_batch 重要參數說明
    # tensors：   排列的張量。
    # batch_size：從隊列中提取新的批量大小。
    # capacity：  一個整數。隊列中元素的最大數量。
    # min_after_dequeue：出隊後隊列中的最小數量元素，用於確保元素的混合級別。

    # ****************************************************************
    # Resize image (Train X)
    image_batch_train = tf.reshape(image_batch, [-1, image_size, image_size, 3])
    # One hot labeling (Trin Y)
    label_batch_train = tf.one_hot(label_batch, Label_size)

    return image_batch_train, label_batch_train


# 定義捲積層function
def conv2d(x, FilterN):
    return tf.layers.conv2d(
        inputs=x,
        filters=FilterN,
        kernel_size=[kernel_size, kernel_size],
        padding='SAME',
        activation=tf.nn.relu)


# 定義池化層function
def max_pool_2x2(x):
    return tf.layers.max_pooling2d(
        inputs=x,
        pool_size=[2, 2],
        strides=2)


def full_connect_layer(x, NeuralNum):
    return tf.layers.dense(inputs=x, units=NeuralNum, activation=tf.nn.relu)


config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

filename = "Paddy_train.tfrecords"
val_File = "Paddy_test.tfrecords"
SavePath = "model\\Paddy.ckpt"
LogPath = "logs\\"

if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("model"):
    os.mkdir("model")

batch_size = 32
data_lens = 1740
minBtch = int(data_lens / batch_size)
val_data_lens = 963
val_minBatch = int(val_data_lens / batch_size)
learning_rate = 1e-4
num_epoch = 500
kernel_size = 5
Label_size = 5
image_size = 128
DoropOutR = 0.7

image_train_batch, label_train_batch = read_and_decode(filename, batch_size)
image_val_batch, label_val_batch = read_and_decode(val_File, batch_size)

x = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
y_ = tf.placeholder(tf.float32, [batch_size, Label_size])
x_image = tf.reshape(x, [-1, image_size, image_size, 3])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('conv1'):
    Output = conv2d(x_image, 32)
    Output = tf.nn.dropout(Output, keep_prob)
    Output = conv2d(Output, 32)
    Output = tf.nn.dropout(Output, keep_prob)

with tf.name_scope('pool1'):
    Output = max_pool_2x2(Output)

with tf.name_scope('conv2'):
    Output = conv2d(Output, 64)
    Output = tf.nn.dropout(Output, keep_prob)
    Output = conv2d(Output, 64)
    Output = tf.nn.dropout(Output, keep_prob)

with tf.name_scope('pool2'):
    Output = max_pool_2x2(Output)

with tf.name_scope('FC'):
    Output = tf.reshape(Output, [-1, 32 * 32 * 64])
    Output = full_connect_layer(Output, 1024)
    Output = tf.nn.dropout(Output, keep_prob)
    Output = full_connect_layer(Output, 1024)
    Output = tf.nn.dropout(Output, keep_prob)

with tf.name_scope('softmax'):
    y_conv = tf.layers.dense(inputs=Output, units=Label_size)
    y_conv = tf.nn.softmax(y_conv)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))
    val_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_conv, labels=y_))

with tf.name_scope('Optimizer'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    val_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    writer_1 = tf.summary.FileWriter(LogPath)
    writer_1.add_graph(sess.graph)
    tf.summary.scalar('acc-train', accuracy)
    tf.summary.scalar('loss-train', cross_entropy)
    tf.summary.scalar('acc-val', val_acc)
    tf.summary.scalar('loss-val', val_loss)
    writer_op = tf.summary.merge_all()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    Best_loss = 100000

    print('Start training...')
    for i in range(num_epoch):

        Stime = time.time()
        for j in range(minBtch):
            image_data, label_data = sess.run([image_train_batch, label_train_batch])
            sess.run(train_step, feed_dict={x: image_data, y_: label_data, keep_prob: DoropOutR})
        Etime = time.time()
        if i % 5 == 0:
            _, train_loss = sess.run([train_step, cross_entropy],
                                     feed_dict={x: image_data, y_: label_data, keep_prob: DoropOutR})
            print('epoch %d, loss : %4.2f, with time : %4.2f s per epoch' % (i, train_loss, (Etime - Stime)))
        if i % 50 == 0:
            tacc = 0
            vacc = 0
            vloss = 0
            for j in range(minBtch):
                image_data, label_data = sess.run([image_train_batch, label_train_batch])
                train_acc = accuracy.eval(feed_dict={x: image_data, y_: label_data, keep_prob: DoropOutR})
                tacc += train_acc
            tacc /= minBtch
            for k in range(val_minBatch):
                image_data, label_data = sess.run([image_val_batch, label_val_batch])
                test_acc, test_loss = sess.run([val_acc, val_loss],
                                               feed_dict={x: image_data, y_: label_data, keep_prob: DoropOutR})
                vacc+=test_acc
                vloss+=test_loss
            vacc/=val_minBatch
            vloss/=val_minBatch
            print('epoch %d, acc : %4.2f%%, val_loss : %4.2f, val_acc : %4.2f%%' % (
                i, tacc * 100, vloss, vacc * 100))
            summary = sess.run(writer_op, feed_dict={x: image_data, y_: label_data, keep_prob: DoropOutR})
            writer_1.add_summary(summary, i)
            writer_1.flush()
            if vloss < Best_loss:
                spath = saver.save(sess, SavePath)
                Best_loss = vloss
                print('Model save with acc : %4.2f%% ' % (vacc * 100))

    print('Model save with acc : %4.2f%% in path %s' % (vacc * 100, spath))
    coord.request_stop()
    coord.join(threads)
print('Done !')
