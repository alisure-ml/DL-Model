import tensorflow as tf
import tensorflow.contrib.slim as tf_slim
from datasets import mnist

dataset = mnist.get_split("train", "C:\ALISURE\DataModel\Data\mnist")

provider = tf_slim.dataset_data_provider.DatasetDataProvider(dataset)

image, label = provider.get(["image", "label"])

print()
