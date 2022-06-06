from re import U
import numpy as np


class DataFactory:

    # 区間の分割の設定
    T = 20
    n = 1000
    h = T / n
    t = np.arange(0, T, h)

    # a = 0.8
    # b = 0.5
    # c = 0.8
    # d = 1.0

    @staticmethod
    def get_lotka_volterra(a,b,c,d):
        n = DataFactory.n
        h = DataFactory.h
        t = DataFactory.t

        # 方程式を定める関数、初期値の定義
        f = lambda u, v, t=0: a * u - b * u * v
        g = lambda u, v, t=0: c * u * v - d * v
        u_0 = 2
        v_0 = 1.1

        # 結果を返すための配列の宣言
        u = np.empty(n)
        v = np.empty(n)
        u[0] = u_0
        v[0] = v_0

        # 方程式を解くための反復計算
        for i in range(n - 1):
            k_1 = h * f(u[i], v[i], t[i])
            k_2 = h * f(u[i] + k_1 / 2, v[i], t[i] + h / 2)
            k_3 = h * f(u[i] + k_2 / 2, v[i], t[i] + h / 2)
            k_4 = h * f(u[i] + k_3, v[i], t[i] + h)
            u[i + 1] = u[i] + 1 / 6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            j_1 = h * g(u[i], v[i], t[i])
            j_2 = h * g(u[i], v[i] + j_1 / 2, t[i] + h / 2)
            j_3 = h * g(u[i], v[i] + j_2 / 2, t[i] + h / 2)
            j_4 = h * g(u[i], v[i] + j_3, t[i] + h)
            v[i + 1] = v[i] + 1 / 6 * (j_1 + 2 * j_2 + 2 * j_3 + j_4)
        return v, u

    @staticmethod
    def get_teacher_data_noisy(a,b,c,d,noise=0.1):
        v, u = DataFactory.get_lotka_volterra(a,b,c,d)
        u_noisy = u + noise * np.std(u) * np.random.randn(u.shape[0])
        v_noisy = v + noise * np.std(v) * np.random.randn(v.shape[0])
        return v_noisy, u_noisy

    @staticmethod
    def get_teacher_data_random(scale=3):
        n = DataFactory.n

        v = np.random.rand(n) * scale
        u = np.random.rand(n) * scale

        return v, u

    @staticmethod
    def get_teacher_data_sincos():

        n = DataFactory.n
        
        u = np.empty(n)
        v = np.empty(n)

        d_theata = 5 * np.pi / n
        alpha = 3 * np.pi / 4
        beta = np.pi / 16
        for i in range(n):
            theata = d_theata * float(i) 
            
            u[i] += np.sin(theata - alpha) + 2.0
            v[i] += np.cos(theata - beta) + 1.3
        
        return v, u