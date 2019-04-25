import os
import cv2
import vgg
import math
import numpy as np
import tensorflow as tf
import vgg_preprocessing
from alisuretool.Tools import Tools
from tensorflow.contrib import slim


def video_to_images(video_filename, image_size=tuple([540, 360])):

    # capture the video
    vid_cap = cv2.VideoCapture(video_filename)
    total_frame = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 可能不准

    # start processing
    print("There are {} frames in the video {}".format(total_frame, video_filename))

    video_frame = []
    while True:
        ret, frame = vid_cap.read()
        if ret:
            frame = cv2.resize(frame, dsize=image_size)
            video_frame.append(frame)
        else:
            break
        pass

    cv2.destroyAllWindows()
    vid_cap.release()

    return video_frame


class InferenceVGG(object):

    def __init__(self, checkpoints_dir='/home/ubuntu/data1.5TB/ImageNetWeights/vgg_16.ckpt'):
        self.checkpoints_file = checkpoints_dir
        self.image_size = 224
        pass

    # 获得预测的结果对应的含义:
    # 修改自：slim.datasets.imagenet.create_readable_names_for_imagenet_labels()
    @staticmethod
    def create_readable_names_for_imagenet_labels():
        synsets_filename = "../slim/datasets/imagenet_lsvrc_2015_synsets.txt"
        metadata_filename = "../slim/datasets/imagenet_metadata.txt"

        synset_list = [s.strip() for s in open(synsets_filename).readlines()]
        assert len(synset_list) == 1000

        synset_to_human_list = open(metadata_filename).readlines()
        assert len(synset_to_human_list) == 21842

        synset_to_human = {}
        for s in synset_to_human_list:
            parts = s.strip().split('\t')
            assert len(parts) == 2
            synset_to_human[parts[0]] = parts[1]

        labels_to_names = {0: 'background'}
        for label_index, synset in enumerate(synset_list):
            labels_to_names[label_index + 1] = synset_to_human[synset]

        return labels_to_names

    @staticmethod
    def _concact_features(conv_output, resize=True):
        if resize:
            conv_output = tf.image.resize_bilinear(conv_output, size=[64, 64])
            pass

        num_or_size_splits = conv_output.get_shape().as_list()[-1]
        each_convs = tf.split(conv_output, num_or_size_splits=num_or_size_splits, axis=3)
        concact_size = int(math.sqrt(num_or_size_splits) / 1)
        all_concact = None
        for i in range(concact_size):
            row_concact = each_convs[i * concact_size]
            for j in range(concact_size - 1):
                row_concact = tf.concat([row_concact, each_convs[i * concact_size + j + 1]], 1)
            if i == 0:
                all_concact = row_concact
            else:
                all_concact = tf.concat([all_concact, row_concact], 2)
        return all_concact

    def build_summaries(self, end_points):
        with tf.name_scope('CNN_outputs'):
            tf.summary.image('images', tf.expand_dims(end_points["images"], 0))
            tf.summary.image('vgg16/1/conv1/conv1_1', self._concact_features(end_points['vgg_16/conv1/conv1_1']))
            tf.summary.image('vgg16/1/conv1/conv1_2', self._concact_features(end_points['vgg_16/conv1/conv1_2']))
            tf.summary.image('vgg16/1/pool1', self._concact_features(end_points['vgg_16/pool1']))

            tf.summary.image('vgg16/2/conv2/conv2_1', self._concact_features(end_points['vgg_16/conv2/conv2_1']))
            tf.summary.image('vgg16/2/conv2/conv2_2', self._concact_features(end_points['vgg_16/conv2/conv2_2']))
            tf.summary.image('vgg16/2/pool2', self._concact_features(end_points['vgg_16/pool2']))

            tf.summary.image('vgg16/3/conv3/conv3_1', self._concact_features(end_points['vgg_16/conv3/conv3_1']))
            tf.summary.image('vgg16/3/conv3/conv3_2', self._concact_features(end_points['vgg_16/conv3/conv3_2']))
            tf.summary.image('vgg16/3/conv3/conv3_3', self._concact_features(end_points['vgg_16/conv3/conv3_3']))
            tf.summary.image('vgg16/3/pool3', self._concact_features(end_points['vgg_16/pool3']))

            tf.summary.image('vgg16/4/conv4/conv4_1', self._concact_features(end_points['vgg_16/conv4/conv4_1']))
            tf.summary.image('vgg16/4/conv4/conv4_2', self._concact_features(end_points['vgg_16/conv4/conv4_2']))
            tf.summary.image('vgg16/4/conv4/conv4_3', self._concact_features(end_points['vgg_16/conv4/conv4_3']))
            tf.summary.image('vgg16/4/pool4', self._concact_features(end_points['vgg_16/pool4']))

            tf.summary.image('vgg16/5/conv5/conv5_1', self._concact_features(end_points['vgg_16/conv5/conv5_1']))
            tf.summary.image('vgg16/5/conv5/conv5_2', self._concact_features(end_points['vgg_16/conv5/conv5_2']))
            tf.summary.image('vgg16/5/conv5/conv5_3', self._concact_features(end_points['vgg_16/conv5/conv5_3']))
            tf.summary.image('vgg16/5/pool5', self._concact_features(end_points['vgg_16/pool5']))
            pass
        pass

    def inference(self, image_filename, log_path):
        with tf.Graph().as_default():
            image = tf.image.decode_jpeg(tf.gfile.FastGFile(image_filename, "rb").read(), channels=3)
            processed_image = vgg_preprocessing.preprocess_image(image, self.image_size, self.image_size, False)
            processed_images = tf.expand_dims(processed_image, 0)

            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, end_points = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
            op_probabilities = tf.nn.softmax(logits)

            end_points["images"] = image
            self.build_summaries(end_points)
            with tf.name_scope("w_b"):
                var_list = tf.global_variables()
                for variable in var_list:
                    tf.summary.histogram(variable.name, variable)
                pass

            op_merged = tf.summary.merge_all()

            init_fn = slim.assign_from_checkpoint_fn(self.checkpoints_file, slim.get_model_variables('vgg_16'))

            with tf.Session() as sess:
                init_fn(sess)

                summary_writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

                probabilities, merged = sess.run([op_probabilities, op_merged])
                probabilities = probabilities[0, 0:]
                sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

                summary_writer.add_summary(merged, 1)

                names = self.create_readable_names_for_imagenet_labels()
                for i in range(5):
                    print('Prob %0.2f%% => [%s]' % (probabilities[sorted_inds[i]] * 100, names[sorted_inds[i] + 1]))

                pass
            pass
        pass

    def feature(self, video_filename, image_size=tuple([224, 224]), pool5_filename=None):
        with tf.Graph().as_default():
            video_frame = video_to_images(video_filename, image_size=image_size)
            images = tf.placeholder(dtype=tf.float32, shape=(None, len(video_frame[0]), len(video_frame[0][0]), 3))

            with slim.arg_scope(vgg.vgg_arg_scope()):
                _, end_points = vgg.vgg_16(images, num_classes=1000, is_training=False, global_pool=True)
                pool5 = end_points["vgg_16/pool5"]
                pass

            init_fn = slim.assign_from_checkpoint_fn(self.checkpoints_file, slim.get_model_variables('vgg_16'))

            with tf.Session() as sess:
                init_fn(sess)
                pool5_results = []
                for index, video_image in enumerate(video_frame):
                    if index % 100 == 0:
                        Tools.print("{} / {}".format(index, len(video_frame)))
                        pass
                    video_image = np.expand_dims(np.float32(video_image - [123.68, 116.779, 103.939]), axis=0)
                    pool5_r = sess.run(pool5, feed_dict={images: video_image})
                    pool5_results.append(pool5_r)
                    pass

                Tools.print("{} OVER".format(len(video_frame)))
                if pool5_filename is not None:
                    Tools.write_to_pkl(pool5_filename, pool5_results)
                pass
            pass
        if pool5_filename is None:
            return pool5_results
        pass

    pass

if __name__ == '__main__':
    inference_vgg = InferenceVGG()
    image_size = tuple([224, 224])

    # image_filename = "20181009_20181009090556_20181009090638_090323"
    # inference_vgg.feature(
    #     video_filename="/home/ubuntu/data1.5TB/video/video_deal/tran/{}.avi".format(image_filename),
    #     image_size=image_size, pool5_filename=Tools.new_dir(
    #         "../pool5/{}_{}_{}.pkl".format(image_filename, image_size[0], image_size[1])))

    image_filename = "20181009_20181009090556_20181009090638_090323"
    pool5_results = inference_vgg.feature(image_size=image_size,
        video_filename="/home/ubuntu/data1.5TB/video/video_deal/tran/{}.avi".format(image_filename))

    image_filename = "20181009_20181009090613_20181009090627_090339"
    pool5_results2 = inference_vgg.feature(image_size=image_size,
        video_filename="/home/ubuntu/data1.5TB/video/video_deal/tran/{}.avi".format(image_filename))

    pool5_results2.extend(pool5_results)
    Tools.write_to_pkl(Tools.new_dir("../pool5/{}_{}_{}.pkl".format(image_filename, image_size[0],
                                                                    image_size[1])), pool5_results2)
    pass
