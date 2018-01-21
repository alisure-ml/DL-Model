import os
from nets import vgg
import tensorflow as tf
from datasets import imagenet
from datasets import dataset_utils
from tensorflow.contrib import slim
from preprocessing import vgg_preprocessing


class InferenceVGG(object):

    def __init__(self, checkpoints_dir='C:\\ALISURE\\DataModel\\Model\\classifier_model\\vgg_16_2016_08_28'):
        self.vgg_16_url = "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz"
        self.checkpoints_dir = checkpoints_dir
        self.image_size = vgg.vgg_16.default_image_size
        self.down_model()
        pass

    # 下载模型
    def down_model(self):
        if not tf.gfile.Exists(self.checkpoints_dir):
            tf.gfile.MakeDirs(self.checkpoints_dir)
            dataset_utils.download_and_uncompress_tarball(self.vgg_16_url, self.checkpoints_dir)
        pass

    def inference(self, image_filename,
                  synsets_filename="C:\ALISURE\DataModel\Data\imagenet\imagenet_lsvrc_2015_synsets.txt",
                  metadata_filename="C:\ALISURE\DataModel\Data\imagenet\imagenet_metadata.txt"):
        with tf.Graph().as_default():
            image_string = tf.gfile.FastGFile(image_filename, "rb").read()
            image = tf.image.decode_jpeg(image_string, channels=3)
            processed_image = vgg_preprocessing.preprocess_image(image, self.image_size, self.image_size, False)
            processed_images = tf.expand_dims(processed_image, 0)

            with slim.arg_scope(vgg.vgg_arg_scope()):
                logits, _ = vgg.vgg_16(processed_images, num_classes=1000, is_training=False)
            probabilities = tf.nn.softmax(logits)

            init_fn = slim.assign_from_checkpoint_fn(os.path.join(self.checkpoints_dir, 'vgg_16.ckpt'),
                                                     slim.get_model_variables('vgg_16'))

            with tf.Session() as sess:
                init_fn(sess)
                np_image, probabilities = sess.run([image, probabilities])
                probabilities = probabilities[0, 0:]
                sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x: x[1])]

            names = imagenet.create_readable_names_for_imagenet_labels(synsets_filename, metadata_filename)
            for i in range(5):
                print('Prob %0.2f%% => [%s]' % (probabilities[sorted_inds[i]] * 100, names[sorted_inds[i] + 1]))
        pass

    pass

if __name__ == '__main__':
    inference_vgg = InferenceVGG()
    inference_vgg.inference(image_filename="../image/demo_5.jpg")
