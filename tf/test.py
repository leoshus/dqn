import numpy as np
import tensorflow as tf

# 手工求导
# X = tf.constant([[1., 2.], [3., 4.]])
# y = tf.constant([[1.], [2.]])
# w = tf.Variable(initial_value=[[1.], [2.]])
# b = tf.Variable(initial_value=1.)
#
# with tf.GradientTape()as tape:
#     L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
# w_grad, b_grad = tape.gradient(L, [w, b])
# print(L, w_grad, b_grad)

X = tf.constant([[1., 2.], [3., 4.]])
y = tf.constant([[1.], [2.]])
w = tf.Variable(initial_value=[[1.], [2.]])
b = tf.Variable(initial_value=1.)

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)
for e in range(num_epoch):
    with tf.GradientTape()as tape:
        L = tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
    grads = tape.gradient(L, [w, b])
    optimizer.apply_gradients(grads_and_vars=zip(grads, [w, b]))
    print(w, b)
