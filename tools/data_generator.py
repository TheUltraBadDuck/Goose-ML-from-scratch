import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from models.model import SupervisedModel, UnsupervisedModel

from tools.data_base import DataBase, ColourList



class GeneratorType:
    
    Nothing = -1
    Linear = 0
    Groups = 1
    Mixture = 100


    @staticmethod
    def make_a_line(start_x: list, end_x: list, y_encode, sample_count: int = 100, feature_var_errors = None):
        
        X = np.linspace(start_x, end_x, sample_count)
        y = np.full(sample_count, y_encode)
        
        if feature_var_errors is None:
            feature_var_errors = np.full(X.shape[1], 1.0, dtype=float)
        
        for i, error in enumerate(feature_var_errors):
            X[:, i] += np.random.normal(0, error, X.shape[0])

        return X, y, GeneratorType.Linear, "Linear Models"


    @staticmethod
    def make_a_group(x, y_encode, sample_count: int = 100, feature_var_errors: list = None, angle: float = 0):

        if (angle != 0) and (len(x) != 2):
            raise Exception("The matrix rotation is impossible with len(x) != 2")

        X = np.full((sample_count, len(x)), x, dtype=float)
        y = np.full(sample_count, y_encode)

        if feature_var_errors is None:
            feature_var_errors = np.full(X.shape[1], 1.0, dtype=float)
        
        modifying_scores = np.zeros(X.shape, dtype=float)
        for i, error in enumerate(feature_var_errors):
            modifying_scores[:, i] = np.random.normal(0, error, X.shape[0])

        if angle != 0:
            rotation_matrix = np.array([
                [np.cos(angle * np.pi / 180), np.sin(angle * np.pi / 180)],
                [-np.sin(angle * np.pi / 180), np.cos(angle * np.pi / 180)]])
            modifying_scores = np.dot(modifying_scores, rotation_matrix)

        X += modifying_scores

        return X, y, GeneratorType.Groups, "Group Models"




class DataGenerator(DataBase):

    def loadData(self, data_reading_model, shuffle: bool = True):

        if self.X is None:
            self.X = data_reading_model[0]
            self.y = data_reading_model[1]
            self.type  = data_reading_model[2]
            self.title = data_reading_model[3]
            self.unique_label = [data_reading_model[1][0]]
            self.limit = [i for i in range(self.X.shape[1])]

        else:
            self.X = np.append(self.X, data_reading_model[0], axis=0)
            self.y = np.append(self.y, data_reading_model[1], axis=0)
            self.unique_label.append(data_reading_model[1][0])

            if self.type != data_reading_model[2]:
                self.type = GeneratorType.Mixture
                self.title = "Mixed Models"

            if shuffle:
                self.shuffle()
  

    def clearData(self):
        self.X, self.y = None, None
        self.type = None
        self.title = ""
        self.limit = None
    

    def shuffle(self):
        p = np.random.permutation(len(self.y))
        self.X = self.X[p]
        self.y = self.y[p]
    


    def drawWhole(self, x = 0, y = 1):
        fig, ax = plt.subplots()

        for i, value in enumerate(self.unique_label):
            mask = np.where(self.y == value, True, False)
            ax.scatter(x=self.X[mask, x],
                        y=self.X[mask, y],
                        color=ColourList[i],
                        marker='o', linewidths=0)

        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")

        fig.suptitle(self.title, fontsize=16, fontweight='bold')

        plt.show()
    

            #ax.text(0.5, 0.5, "Unsupervised model does not have a training model", fontsize=12, fontstyle='italic', va="center", ha="center")



    def drawResult(self, x = 0, y = 1):
        fig, ax = plt.subplots()

        for i, value in enumerate(self.unique_label):
            mask = np.where(self.y == value, True, False)
            ax.scatter(x=self.X[mask, x],
                        y=self.X[mask, y],
                        color=ColourList[i],
                        marker='o', linewidths=0)
        
        if issubclass(type(self.model), SupervisedModel):
            pass
        else:
            self.model.makePlot(self.X, x, y, self.X_transform, ax)
        
        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")

        fig.suptitle("Linear data", fontsize=16, fontweight='bold')

        plt.show()


