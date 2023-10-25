from models.model import UnsupervisedModel
from tools.load_data         import LoadData
from tools.generate_data     import GenerateData
from tools.custom_graph_data import CustomGraphData

from tools.activ_func import *

from models.knn                  import KNN
from models.linear_regression    import LinearRegression
from models.logistic_regression  import LogisticRegression
from models.naive_bayes          import NaiveBayes
from models.perceptron           import Perceptron
from models.svm                  import SVM
from models.decision_tree        import DecisionTree
from models.neural_network       import NeuralNetwork
from models.pca                  import PCA
from models.k_means              import KMeansClustering



import matplotlib.pyplot as plt


def main():
    # model_obj = GenerateData(type="blobs", n_samples=500, n_features=2, centers=3, noise=5)
    # #model_obj = LoadData(type="breast_cancer", limit=[0, 1, 2, 4, 5])
    # #model_obj.split(test_size=0.2, random_state=1234)
    # #model_obj.compare()
    # #model_obj.setModel(SVM(learning_rate=0.002, n_iters=1000))
    # #model_obj.draw()
    # model_obj.setModel(KMeansClustering(iters=25, k=2))
    # model_obj.train()
    # model_obj.transform()
    # model_obj.checkAccuracy()
    # #model_obj.test()
    # #model_obj.checkAccuracy()

    def gr(x):
        if x > 6:
            return x
        elif x > 4:
            return 0.5*x + 3
        else:
            return 0.25*x + 4


    model_obj = CustomGraphData()
    model_obj.loadData(lambda x: gr(x), 0, 10, 1000, 0.1)
    model_obj.split(test_size=0.2)

    model_obj.setModel(NeuralNetwork(learning_rate=0.05, n_iters=30000,
                                     hidden_nodes=[2, 3],
                                     activ_func=[ReLU, ReLU, Linear],
                                     deriv_func=[ReLU_Deriv, ReLU_Deriv, Linear_Deriv],
                                     printing=False, running_time=10))
    model_obj.train()
    model_obj.test()
    model_obj.checkAccuracy()

    # model_obj = LoadData("iris")
    # model_obj.setModel(PCA(n_components=2))
    # model_obj.train()
    # model_obj.transform()
    # model_obj.checkAccuracy(type="cluster")






if __name__ == "__main__":
    main()


