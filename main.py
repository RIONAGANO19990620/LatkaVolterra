from DataFactory import DataFactory
from NeuralNet import PhysicsInformedNN


def main():
    v_noisy, u_noisy = DataFactory.get_teacher_data()
    v, u = DataFactory.get_lotka_volterra()
    t_array = DataFactory.t
    nn = PhysicsInformedNN(t_array, u_noisy, v_noisy)
    nn.train(10000)
    nn.compare_numerical_ans(u, v)


if __name__ == '__main__':
    main()
