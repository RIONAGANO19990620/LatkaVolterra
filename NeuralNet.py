import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.layers import Dense, Input
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

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

    def __reduce_lr(self) -> ReduceLROnPlateau:
        return ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=10,
            min_lr=0.0001
        )

    def train(self, epochs=100):
        u_data = self.normalized_data_list.u_normalized.array
        v_data = self.normalized_data_list.v_normalized.array
        self.history = self.model.fit(self.normalized_data_list.t_normalized.data,
                                    #  y={"u_data": u_data, "v_data": v_data}, epochs=epochs,
                                    #   callbacks=[self.__reduce_lr()])
                                      y={"u_data": u_data, "v_data": v_data}, epochs=epochs)

    def plot_coefficient(self):
        plt.plot(self.history.history['a'], label='a')
        plt.plot(self.history.history['b'], label='b')
        plt.plot(self.history.history['c'], label='c')
        plt.plot(self.history.history['d'], label='d')
        plt.ylim(0, 1.1)
        plt.legend()
        plt.xlabel('train_num')
        plt.savefig('coefficient.png')
        plt.close()
        plt.clf()

        # 係数の推論履歴から一番最後のものを取り出し、推論結果として返す
        a = float(self.history.history['a'][-1][0])
        b = float(self.history.history['b'][-1][0])
        c = float(self.history.history['c'][-1][0])
        d = float(self.history.history['d'][-1][0])

        return a,b,c,d

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

        return  u_predict, v_predict

    @staticmethod
    def plot_prediction_result(u_teacher, v_teacher, u_predict, v_predict, u_predict2, v_predict2):

        plt.plot(u_teacher, linestyle='--', label='teacher data', color='red')
        plt.plot(v_teacher, linestyle='--', color='red')

        plt.plot(u_predict, linestyle='-.', label='prediction result', color='blue')
        plt.plot(v_predict, linestyle='-.', color='blue')

        plt.plot(u_predict2, linestyle='--', label='prediction result(using predicted coeff)', color='green')
        plt.plot(v_predict2, linestyle='--', color='green')

        plt.legend()
        plt.savefig('prediction.png')
        plt.close()
        plt.clf()

    def print_coeffisient(self):
        print('a={0}, b={1}, c={2}, d={3}'.format(self.model.a.numpy(), self.model.b.numpy(), self.model.c.numpy(),
                                                  self.model.d.numpy()))
