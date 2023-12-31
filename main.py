from models.model import UnsupervisedModel
from tools.data_loader       import DataLoader
from tools.data_generator    import DataGenerator, GeneratorType
from tools.data_custom_graph import DataCustomGraph

from tools.activ_func import *

from models.knn                  import KNN
from models.linear_regression    import LinearRegression
from models.logistic_regression  import LogisticRegression
from models.softmax_regression   import SoftmaxRegression
from models.naive_bayes          import NaiveBayes
from models.perceptron           import Perceptron
from models.svm                  import SVM
from models.decision_tree        import DecisionTree
from models.neural_network       import NeuralNetwork
from models.pca                  import PCA
from models.kmeans               import KMeans


from sklearn.datasets import *


def main():

    model_obj = DataGenerator()
    #model_obj.loadData(GeneratorType.make_a_line([0, 0], [100, 50], 0, 300, [8, 8]))
    #model_obj.loadData(GeneratorType.make_a_group([20, 40], 1, 180, [13, 3], 30))
    #model_obj.loadData(GeneratorType.make_a_group([80, 10], 2, 180, [13, 3], 30))
    #model_obj.loadData(GeneratorType.make_a_roll([6, 3], 0, 500, rotating_time=2, expanding=15, angle=90, clockwise=True))
    #model_obj.loadData(GeneratorType.make_a_roll([6, 3], 1, 500, rotating_time=2, expanding=15, angle=270, clockwise=True))
    
    model_obj.loadData(GeneratorType.make_a_donut([0, 0], 0, 135, [0.5, 0.5], 7, -150, -30))
    model_obj.loadData(GeneratorType.make_a_donut([-6, 8], 1, 90, [0.5, 0.5], 3, 0, 360))
    model_obj.loadData(GeneratorType.make_a_donut([6, 8], 2, 90, [0.5, 0.5], 3, 0, 360))

    model_obj.loadData(GeneratorType.make_a_line([0, 0], [0, 5], 3, 60, [0.5, 0.5]))

    model_obj.drawWhole()

    model_obj.setAndLearnModel(KMeans(k=3))
    model_obj.drawResult()

    # model_obj.setAndLearnModel(KNN(k=5), "c", 0.2)
    # model_obj.drawResult(graph_type="scatter")
    # model_obj.drawResult(graph_type="matrix")




    # model_obj = GenerateData(type="blobs", n_samples=500, n_features=2, centers=3, noise=5)
    # #model_obj = LoadData(type="breast_cancer", limit=[0, 1, 2, 4, 5])
    # #model_obj.split(test_size=0.2, random_state=1234)
    # #model_obj.compare()
    # #model_obj.setModel(SVM(learning_rate=0.002, n_iters=1000))
    # #model_obj.draw()
    # model_obj.setModel(KMeansClustering(iters=25, k=2))
    # model_obj.train()
    # model_obj.transform()
    # model_obj.drawTest()
    # #model_obj.test()
    # #model_obj.drawTest()

    # def gr(x):
    #     if x > 6:
    #         return 1.5*x - 3
    #     elif x > 4:
    #         return 0.5*x + 3
    #     else:
    #         return 0.25*x + 4

    # def gr(x):
    #     return abs(0.6 * abs(1.4 * x - 8) - 1)


    # model_obj = CustomGraphData()
    # model_obj.loadData(lambda x: gr(x), 0, 10, 500, 0.2)
    # model_obj.split(test_size=0.2)

    # model_obj.draw()

    # model_obj.setModel(NeuralNetwork(learning_rate=0.01, n_iters=15000,
    #                                  hidden_nodes=[3, 5, 3],
    #                                  activ_func=[ReLU, ReLU, ReLU, Linear],
    #                                  deriv_func=[ReLU_Deriv, ReLU_Deriv, ReLU_Deriv, Linear_Deriv],
    #                                  printing=False, running_time=5))
    # model_obj.train()
    # model_obj.test()
    # model_obj.drawTest()

    # model_obj = LoadData("wine", [0, 1, 2, 4])
    # model_obj.split(0.2)

    # model_obj.setModel(SoftmaxRegression(learning_rate=0.05, n_iters=1000, optimized_softmax=True))
    # model_obj.compare()
    # model_obj.train()
    # model_obj.test()

    # model_obj.drawTest(graph_type="region", feature_x=0, feature_y=4)

    #model_obj.transform()
    #model_obj.drawTest()






if __name__ == "__main__":
    main()


