import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Input

from Model import PPINs
from NormalizedData import NormalizedDataList


class PhysicsInformedNN:

    def __init__(self, t_array, u_data, v_data):
        self.normalized_data_list = NormalizedDataList(u_data, v_data, t_array)
        self.__initial_NN()

    def __initial_NN(self):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            t = Input(shape=(1,))
            z = Dense(1, activation="tanh")(t)
            for _ in range(8):
                z = Dense(20, activation='tanh')(z)
            z = Dense(2, activation="linear")(z)

            self.model = PPINs(self.normalized_data_list, t, z)
            self.model.compile(optimizer="adam", metrics=['loss', 'mae', 'a', 'b'])

    def train(self, epochs=100):
        u_data = self.normalized_data_list.u_normalized.array
        v_data = self.normalized_data_list.v_normalized.array
        self.history = self.model.fit(self.normalized_data_list.t_normalized.data,
                                      y={"u_data": u_data, "v_data": v_data}, epochs=epochs)

    def plot_coefficient(self):
        plt.plot(self.history.history['a'], label='a')
        plt.plot(self.history.history['b'], label='b')
        plt.plot(self.history.history['c'], label='c')
        plt.plot(self.history.history['d'], label='d')
        plt.xlabel('train_num')
        plt.savefig('coefficient.png')
        plt.close()
        plt.clf()

    def compare_numerical_ans(self, u, v):
        predict = self.model.predict(self.normalized_data_list.t_normalized.data)
        u_predict = predict[:, 0]
        v_predict = predict[:, 1]
        plt.plot(u, linestyle='--', label='numerical calculation', color='red')
        plt.plot(u_predict, linestyle='-.', label='neural network', color='blue')
        plt.plot(v, linestyle='--', color='red')
        plt.plot(v_predict, linestyle='-.', color='blue')
        plt.legend()
        plt.savefig('compare.png')
        plt.close()
        plt.clf()

    def print_coeffisient(self):
        print('a={0}, b={1}, c={2}, d={3}'.format(self.model.a.numpy(), self.model.b.numpy(), self.model.c.numpy(),
                                                  self.model.d.numpy()))
