import tensorflow as tf
import numpy as np
from tensorflow.keras import Model, layers, optimizers

X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
y = tf.constant([[10.0], [20.0]])


# 线性回归问题DEMO
class Linear(Model):
    def __init__(self):
        super().__init__()
        self.dense = layers.Dense(
            units=1,  # 输出张量纬度
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        output = self.dense(input)
        return output


model = Linear()
optimizer = optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape()as tape:
        y_pred = model(X)
        loss = tf.reduce_sum(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

print(model.variables)
