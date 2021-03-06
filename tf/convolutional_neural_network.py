import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, Model, layers, losses, optimizers, metrics


# 卷积神经网络DEMO
class CNN(Model):
    def __init__(self):
        super().__init__()
        self.conv1 = layers.Conv2D(
            filters=32,  # 卷积层神经元(卷积核)数目
            kernel_size=[5, 5],  # 感受野大小
            padding='same',  # padding策略(valid或same)
            activation=tf.nn.relu  # 激活函数
        )
        self.pool1 = layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)  # [batch_size, 28, 28, 32]
        x = self.pool1(x)  # [batch_size, 14, 14, 32]
        x = self.conv2(x)  # [batch_size, 14, 14, 64]
        x = self.pool2(x)  # [batch_size, 7, 7, 64]
        x = self.flatten(x)  # [batch_size, 7 * 7 * 64]
        x = self.dense1(x)  # [batch_size, 1024]
        x = self.dense2(x)  # [batch_size, 10]
        output = tf.nn.softmax(x)
        return output
