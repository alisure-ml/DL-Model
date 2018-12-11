import os
import vgg
import math
import tensorflow as tf
from tensorflow.contrib import slim
import vgg_preprocessing


class InferenceVGG(object):

    def __init__(self, checkpoints_dir='/home/ubuntu/PycharmProjects/ALISURE/attention/RelateByMaskAttention/pretrained_model/vgg_16.ckpt'):
        self.checkpoints_file = checkpoints_dir
        self.image_size = 224

        self.synsets_filename = "../slim/datasets/imagenet_lsvrc_2015_synsets.txt"
        self.metadata_filename = "../slim/datasets/imagenet_metadata.txt"
        pass

    # 获得预测的结果对应的含义:
    # 修改自：slim.datasets.imagenet.create_readable_names_for_imagenet_labels()
    def create_readable_names_for_imagenet_labels(self):
        synset_list = [s.strip() for s in open(self.synsets_filename).readlines()]
        assert len(synset_list) == 1000

        synset_to_human_list = open(self.metadata_filename).readlines()
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

    @staticmethod
    def _concact_features(conv_output, resize=False):
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

    pass

if __name__ == '__main__':
    inference_vgg = InferenceVGG()
    inference_vgg.inference(image_filename="../image/demo_2.jpg", log_path="../log")
    # inference_vgg.inference_dir(image_dir="../image")
