import os
import vgg
import tensorflow as tf
from tensorflow.contrib import slim
import vgg_preprocessing


class InferenceVGG(object):

    def __init__(self, checkpoints_dir='C:\\ALISURE\\DataModel\\Model\\classifier_model\\vgg_16_2016_08_28\\vgg_16.ckpt'):
        self.checkpoints_file = checkpoints_dir
        self.image_size = 224

        self.synsets_filename = "C:\\ALISURE\\DataModel\\Data\\imagenet\\imagenet_lsvrc_2015_synsets.txt"
        self.metadata_filename = "C:\\ALISURE\\DataModel\\Data\\imagenet\\imagenet_metadata.txt"
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

    def inference(self, image_filename):
        with tf.Graph().as_default():
            image = tf.image.decode_jpeg(tf.gfile.FastGFile(image_filename, "rb").read(), channels=3)
            processed_image = vgg_preprocessing.preprocess_image(image, self.image_size, self.image_size, False)
            processed_images = tf.expand_dims(processed_image, 0)

            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
            probabilities = tf.nn.softmax(logits)

            init_fn = slim.assign_from_checkpoint_fn(self.checkpoints_file, slim.get_model_variables('vgg_16'))

            with tf.Session() as sess:
                init_fn(sess)
                probabilities = sess.run(probabilities)
                probabilities = probabilities[0, 0:]
                sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

            names = self.create_readable_names_for_imagenet_labels()
            for i in range(5):
                print('Prob %0.2f%% => [%s]' % (probabilities[sorted_inds[i]] * 100, names[sorted_inds[i] + 1]))
        pass

    def inference_dir(self, image_dir):
        # 得到所有需要预测的图片
        images_filename = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]

        with tf.Graph().as_default():
            # 构建推理图
            op_inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3])
            with slim.arg_scope(vgg.vgg_arg_scope()):
                op_logits, _ = vgg.vgg_16(op_inputs, num_classes=1000, is_training=False)
            op_probabilities = tf.nn.softmax(op_logits)

            # 恢复模型
            init_fn = slim.assign_from_checkpoint_fn(self.checkpoints_file, slim.get_model_variables('vgg_16'))

            with tf.Session() as sess:
                init_fn(sess)

                for image_filename in images_filename:
                    # 读取数据
                    image = tf.image.decode_jpeg(tf.gfile.FastGFile(image_filename, "rb").read(), channels=3)
                    processed_image = vgg_preprocessing.preprocess_image(image, self.image_size, self.image_size, False)
                    processed_images = tf.expand_dims(processed_image, 0)
                    images = sess.run(processed_images)
                    # 预测
                    probabilities = sess.run(op_probabilities, feed_dict={op_inputs: images})
                    probabilities = probabilities[0, 0:]
                    sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

                    # 转换并打印
                    names = self.create_readable_names_for_imagenet_labels()
                    for i in range(5):
                        print('Prob %0.2f%% => [%s]' % (probabilities[sorted_inds[i]] * 100, names[sorted_inds[i] + 1]))
                    print()
                pass
        pass

    pass

if __name__ == '__main__':
    inference_vgg = InferenceVGG()
    inference_vgg.inference(image_filename="../image/demo_5.jpg")
    inference_vgg.inference_dir(image_dir="../image")
