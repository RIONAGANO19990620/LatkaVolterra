import numpy as np


class DataFactory:
    # 区間の分割の設定
    T = 20
    n = 1000
    h = T / n
    t = np.arange(0, T, h)

    a = 0.8
    b = 0.5
    c = 0.8
    d = 1.0

    # 方程式を定める関数、初期値の定義
    f = lambda u, v, t=0: DataFactory.a * u - DataFactory.b * u * v
    g = lambda u, v, t=0: DataFactory.c * u * v - DataFactory.d * v
    u_0 = 2
    v_0 = 1.1

    @staticmethod
    def get_lotka_volterra():
        n = DataFactory.n
        u_0 = DataFactory.u_0
        v_0 = DataFactory.v_0
        h = DataFactory.h
        f = DataFactory.f
        g = DataFactory.g
        t = DataFactory.t
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
    def get_teacher_data(noise=0.1):
        v, u = DataFactory.get_lotka_volterra()
        u_noisy = u + noise * np.std(u) * np.random.randn(u.shape[0])
        v_noisy = v + noise * np.std(v) * np.random.randn(v.shape[0])
        return v_noisy, u_noisy