import tensorflow as tf
import numpy as np
import cv2
import os
import sklearn.metrics
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_File(file_dir):
    # The images in each subfolder
    images = []
    # The subfolders
    subfolders = []

    # Using "os.walk" function to grab all the files in each folder
    for dirPath, dirNames, fileNames in os.walk(file_dir):
        for name in fileNames:
            images.append(os.path.join(dirPath, name))
        for name in dirNames:
            subfolders.append(os.path.join(dirPath, name))

    # To record the labels of the image dataset
    labels = []
    count = 0
    for a_folder in subfolders:
        n_img = len(os.listdir(a_folder))
        labels = np.append(labels, n_img * [count])
        count += 1

    subfolders = np.array([images, labels])
    # subfolders = subfolders.transpose()
    subfolders = subfolders[:, np.random.permutation(subfolders.shape[1])].T

    image_list = list(subfolders[:, 0])
    label_list = list(subfolders[:, 1])
    label_list = [int(float(i)) for i in label_list]
    return image_list, label_list


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


def cv_load_image(Path):
    if "shi10" in Path:
        cvLabel = 0
    elif "tc192" in Path:
        cvLabel = 1
    elif "tk9" in Path:
        cvLabel = 3
    elif "tk14" in Path:
        cvLabel = 2
    elif "tn11" in Path:
        cvLabel = 4
    image = cv2.imread(Path)
    height, width = image.shape[:2]
    if height != width:
        BorderSize = 0
        if (height > width):
            BorderSize = height
        else:
            BorderSize = width
        image = cv2.copyMakeBorder(image, int((BorderSize - height) / 2), int((BorderSize - height) / 2)
                                   , int((BorderSize - width) / 2), int((BorderSize - width) / 2),
                                   cv2.BORDER_CONSTANT)
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_CUBIC)
    b, g, r = cv2.split(image)
    rgb_image = cv2.merge([r, g, b])
    rgb_image = tf.cast(rgb_image, tf.float32) * (1. / 255)
    tf_image = tf.reshape(rgb_image, [-1, image_size, image_size, 3])
    return tf_image, cvLabel


def show_feature_map(layer, layer_name, num_or_size_splits, axis, max_outputs):
    split = tf.split(layer, num_or_size_splits=num_or_size_splits, axis=axis)
    for i in range(num_or_size_splits):
        tf.summary.image(layer_name + "-" + str(i), split[i], max_outputs)


config = tf.ConfigProto(allow_soft_placement=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
config.gpu_options.allow_growth = True

SingleImage = False
filename = "Paddy_train.tfrecords"
val_File = "Paddy_val.tfrecords"
SavePath = "model\\Paddy.ckpt"
LogPath = "logs\\"
ImagePath = "test\\Paddy\\tn11\\361.bmp"
DataDir = "validation"

# ImageList, _ = get_File(DataDir)

if not os.path.isdir("logs"):
    os.mkdir("logs")
if not os.path.isdir("model"):
    os.mkdir("model")

batch_size = 32
data_lens = 1740
minBtch = int(data_lens / batch_size)
val_data_lens = 5612
val_minBatch = int(val_data_lens / batch_size)
# val_minBatch = minBtch
learning_rate = 1e-4
num_epoch = 1000
kernel_size = 5
Label_size = 5
image_size = 128
LogPath = "logs\\test\\"
if not os.path.isdir(LogPath):
    os.mkdir(LogPath)

if SingleImage:
    load_image, load_label = cv_load_image(ImagePath)
    batch_size = 1
else:
    image_val_batch, label_val_batch = read_and_decode(val_File, batch_size)
    batch_size = 32

x = tf.placeholder(tf.float32, [batch_size, image_size, image_size, 3])
y_ = tf.placeholder(tf.float32, [batch_size, Label_size])
x_image = tf.reshape(x, [-1, image_size, image_size, 3])
keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('conv1'):
    Output = conv2d(x_image, 32)
    show_feature_map(layer=Output, layer_name="conv1-1", num_or_size_splits=32, axis=3, max_outputs=3)
    Output = conv2d(Output, 32)
    show_feature_map(layer=Output, layer_name="conv1-2", num_or_size_splits=32, axis=3, max_outputs=3)

with tf.name_scope('pool1'):
    Output = max_pool_2x2(Output)

with tf.name_scope('conv2'):
    Output = conv2d(Output, 64)
    show_feature_map(layer=Output, layer_name="conv2-1", num_or_size_splits=64, axis=3, max_outputs=3)
    Output = conv2d(Output, 64)
    show_feature_map(layer=Output, layer_name="conv2-2", num_or_size_splits=64, axis=3, max_outputs=3)

with tf.name_scope('pool2'):
    Output = max_pool_2x2(Output)

with tf.name_scope('FC'):
    Output = tf.reshape(Output, [-1, 32 * 32 * 64])
    Output = full_connect_layer(Output, 1024)
    Output = full_connect_layer(Output, 1024)

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

saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "model\\Paddy.ckpt")
    # #***************************************************
    # writer_1 = tf.summary.FileWriter(LogPath)
    # writer_1.add_graph(sess.graph)
    # writer_op = tf.summary.merge_all()
    # writer_1.flush()
    # #***************************************************

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    print('Start...')

    # ****************************************************************************
    if SingleImage:
        accCount = 0
        ImageCount = 0
        for f in ImageList:
            try:
                Stime = time.time()
                load_image, load_label = cv_load_image(f)
                image_data = sess.run(load_image)
                predict = y_conv.eval(feed_dict={x: image_data})
                Etime = time.time()
                predictIndex = list(np.argmax(predict, 1))
                if predictIndex[0] == load_label:
                    accCount += 1
                    print("Predict : %d , Answer : %d , Time : %f sec, v" % (
                        predictIndex[0], load_label, (Etime - Stime)))
                else:
                    print("Predict : %d , Answer : %d , Time : %f sec, x" % (
                        predictIndex[0], load_label, (Etime - Stime)))
                ImageCount += 1
                # summary = sess.run(writer_op, feed_dict={x: image_data})
                # writer_1.add_summary(summary, i)
                # writer_1.flush()
            except:
                print(f + ", image incorrect...")
        print("Accuracy %4.2f%%" % (accCount / ImageCount))
    # ****************************************************************************
    else:
        test_acc = 0
        for i in range(val_minBatch):
            image_data, label_data = sess.run([image_val_batch, label_val_batch])
            acc = accuracy.eval(feed_dict={x: image_data, y_: label_data})
            test_acc += acc
            sess.run(y_conv, feed_dict={x: image_data})
            predict = y_conv.eval(feed_dict={x: image_data})
            if i == 0:
                predictIndex = list(np.argmax(predict, 1))
                label_dataIndex = list(np.argmax(label_data, 1))
            else:
                predictIndex.extend(list(np.argmax(predict, 1)))
                label_dataIndex.extend(list(np.argmax(label_data, 1)))

        test_acc /= val_minBatch

        # for j in range(batch_size):
        #     print(predict[j], label_data[j])
        print('Test accuracy : %4.2f%%' % (test_acc * 100))
        print(sklearn.metrics.confusion_matrix(label_dataIndex, predictIndex, labels=range(Label_size)))

    coord.request_stop()
    coord.join(threads)

print('Done !')
