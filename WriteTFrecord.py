import tensorflow as tf
import cv2
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ******************************************************************
# 轉Int64資料為 tf.train.Feature 格式
def int64_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


# 轉Bytes資料為 tf.train.Feature 格式
def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


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


def convert_to_TFRecord(images, labels, filename):
    n_samples = len(labels)
    print("Convert " + str(n_samples) + " samples")
    TFWriter = tf.python_io.TFRecordWriter(filename)

    print('\nTransform start...')
    ConvertCount = 0
    for i in np.arange(0, n_samples):
        try:
            image = cv2.imread(images[i])

            if image is None:
                print('Error image:' + images[i])
            else:
                #
                # height, width = image.shape[:2]
                # if height != width:
                #     BorderSize = 0
                #     if (height > width):
                #         BorderSize = height
                #     else:
                #         BorderSize = width
                #     nimg = cv2.copyMakeBorder(image, int((BorderSize - height) / 2), int((BorderSize - height) / 2)
                #                               , int((BorderSize - width) / 2), int((BorderSize - width) / 2),
                #                               cv2.BORDER_CONSTANT)

                image = cv2.resize(image, (cImageSize, cImageSize), cv2.INTER_CUBIC)
                b, g, r = cv2.split(image)
                rgb_image = cv2.merge([r, g, b])
                image_raw = rgb_image.tostring()
                # image_np = np.array(image, dtype=np.uint8)
                # image_np = image_np.astype('float32')
                # image_np = np.multiply(image_np, 1.0 / 255.0)
                # image_raw = image_np.tobytes()

            label = int(labels[i])

            # 將 tf.train.Feature 合併成 tf.train.Features
            ftrs = tf.train.Features(
                feature={'Label': int64_feature(label),
                         'image_raw': bytes_feature(image_raw)}
            )

            # 將 tf.train.Features 轉成 tf.train.Example
            example = tf.train.Example(features=ftrs)

            # 將 tf.train.Example 寫成 tfRecord 格式
            TFWriter.write(example.SerializeToString())
            ConvertCount += 1
        except IOError as e:
            print('Skip!\n')

    TFWriter.close()
    print('Transform %d images done!' % (ConvertCount))


# # 將image data存為tfRecord格式
train_dataset_dir = 'train\\Paddy'
cImageSize = 128
images, labels = get_File(train_dataset_dir)

convert_to_TFRecord(images, labels, 'Paddy_train.tfrecords')
