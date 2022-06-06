from DataFactory import DataFactory
from NeuralNet import PhysicsInformedNN
import math


def main():

    # 初期値設定
    a = 0.8
    b = 0.35
    c = 0.65
    d = 1.0

    # Latoka-Volterraの理論解
    v_correct, u_correct = DataFactory.get_lotka_volterra(a, b, c, d)

    # 教師データ作成
    v_teacher, u_teacher = DataFactory.get_teacher_data_noisy(a, b, c, d)
    # v_teacher, u_teacher = DataFactory.get_teacher_data_random()
    # v_teacher, u_teacher = DataFactory.get_teacher_data_sincos() 
    t_array = DataFactory.t

    # 学習・予測
    nn = PhysicsInformedNN(t_array, u_teacher, v_teacher)
    nn.train(10000)
    a_predict,b_predict,c_predict,d_predict = nn.plot_coefficient()
    u_predict, v_predict = nn.compare_numerical_ans(u_correct, v_correct)

    v_predict2, u_predict2 = DataFactory.get_lotka_volterra(a=a_predict, b=b_predict, c=c_predict, d=d_predict)
    PhysicsInformedNN.plot_prediction_result(u_teacher, v_teacher, u_predict, v_predict,  u_predict2, v_predict2)
    
    print("推論結果　a: {0}, b: {1}, c: {2}, d: {3}".format(a_predict,b_predict,c_predict,d_predict))
    print("aの誤差: {0}, bの誤差: {1}, cの誤差: {2}, dの誤差: {3}".format(a_predict-a, b_predict-b, c_predict-c, d_predict-d))



if __name__ == '__main__':
    main()
