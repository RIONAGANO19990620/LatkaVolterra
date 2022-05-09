import tensorflow as tf
from tensorflow import keras

from NormalizedData import NormalizedDataList

loss_tracker = keras.metrics.Mean(name="loss")
mae_metric = keras.metrics.MeanAbsoluteError(name="mae")


class PPINs(keras.Model):

    def __init__(self, normalized_data_list: NormalizedDataList, *args, **kwargs):
        super(PPINs, self).__init__(*args, **kwargs)
        self.a = tf.Variable([0], dtype=tf.float32)
        self.b = tf.Variable([0], dtype=tf.float32)
        self.c = tf.Variable([0], dtype=tf.float32)
        self.d = tf.Variable([0], dtype=tf.float32)
        self.normalized_data_list = normalized_data_list

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = tf.reduce_mean(tf.reduce_mean(tf.square(y['u_data'] - y_pred[:, 0]))) + tf.reduce_mean(
                tf.square(y['v_data'] - y_pred[:, 1])) + tf.reduce_mean(
                tf.square(PPINs.pred(y_pred, x, self.a, self.b, self.c, self.d, self.normalized_data_list)))
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        loss_tracker.update_state(loss)
        mae_metric.update_state(y['u_data'], y_pred[:, 0])
        mae_metric.update_state(y['v_data'], y_pred[:, 1])
        return {"loss": loss_tracker.result(), "mae": mae_metric.result(), 'a': self.a, "b": self.b, 'c': self.c,
                'd': self.d}

    @staticmethod
    def pred(y_pred, x, a, b, c, d, normalized_data_list: NormalizedDataList):
        u = y_pred[:, 0]
        v = y_pred[:, 1]
        u_t = tf.gradients(u, x)
        v_t = tf.gradients(v, x)
        u_m = normalized_data_list.u_normalized.max - normalized_data_list.u_normalized.min
        v_m = normalized_data_list.v_normalized.max - normalized_data_list.v_normalized.min
        t_m = normalized_data_list.t_normalized.max - normalized_data_list.t_normalized.min
        u = tf.add(tf.multiply(u, u_m),
                   normalized_data_list.u_normalized.min)
        v = tf.add(tf.multiply(v, v_m),
                   normalized_data_list.v_normalized.min)
        u_t = tf.divide(tf.multiply(u_t, u_m), t_m)
        return v_t - c * u * v + d * v + u_t - a * u + b * u * v
