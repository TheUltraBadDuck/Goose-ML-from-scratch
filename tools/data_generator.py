import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from models.model import SupervisedModel, UnsupervisedModel

from tools.data_base import DataBase, ColourList



class GeneratorType:
    
    Nothing = -1
    Linear = 0
    Groups = 1
    Spiral = 2
    Mixture = 100
        

    @staticmethod
    def make_a_line(start_x: list, end_x: list, colour_val, sample_count: int = 100, feature_var_errors: list = None):
        
        X = np.linspace(start_x, end_x, sample_count)
        colours = np.full(sample_count, colour_val)

        if feature_var_errors is None:
            feature_var_errors = np.full(X.shape[1], 1.0, dtype=float)
        
        for i, error in enumerate(feature_var_errors):
            X[:, i] += np.random.normal(0, error, X.shape[0])

        return X, colours, GeneratorType.Linear, "Linear Models"


    @staticmethod
    def make_a_group(x, colour_val, sample_count: int = 100, feature_var_errors: list = None, angle: float = 0):

        if (angle != 0) and (len(x) != 2):
            raise Exception("The matrix rotation is impossible with len(x) != 2")

        X = np.full((sample_count, len(x)), x, dtype=float)
        colours = np.full(sample_count, colour_val)

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

        return X, colours, GeneratorType.Groups, "Group Models"

    
    @staticmethod
    def make_a_roll(x, colour_val, sample_count: int = 100, feature_var_errors: list = None,
                    angle: float = 0.0, rotating_time: float = 1.5, expanding: float = 10, clockwise: bool = False):

        if (angle != 0) and (len(x) != 2):
            raise Exception("The matrix rotation is impossible with len(x) != 2")

        # Make a horizontal line from (0, 0) to (expanding * rotating_time, 0)
        X = np.zeros((sample_count, len(x)), dtype=float)
        colours = np.full(sample_count, colour_val)

        # Make a spiral starting from angle 0
        if feature_var_errors is None:
            feature_var_errors = np.full(X.shape[1], 1.0, dtype=float)
        
        modifying_scores = np.zeros(X.shape, dtype=float)
        modifying_scores[:, 0] += np.linspace(0, expanding * rotating_time, sample_count, dtype=float)
        
        angle_group = np.linspace(0, 360 * rotating_time, sample_count)
        if clockwise:
            angle_group = angle_group[::-1]
        for i in range(sample_count):
            new_angle = angle_group[i] + angle
            rotation_matrix = np.array([[np.cos(new_angle * np.pi / 180), np.sin(new_angle * np.pi / 180)],
                                        [-np.sin(new_angle * np.pi / 180), np.cos(new_angle * np.pi / 180)]])

            modifying_scores[i, :] = np.dot(modifying_scores[i, :].reshape(1, 2), rotation_matrix).reshape(-1)
        
        # Add noise
        for i, error in enumerate(feature_var_errors):
            modifying_scores[:, i] += np.random.normal(0, error, X.shape[0])

        X += modifying_scores

        return X, colours, GeneratorType.Spiral, "Spiral Models"



class DataGenerator(DataBase):

    def loadData(self, data_reading_model, shuffle: bool = True):

        if self.X is None:
            self.X = data_reading_model[0]
            self.colours = data_reading_model[1]
            self.type  = data_reading_model[2]
            self.title = data_reading_model[3]
            self.unique_label = [data_reading_model[1][0]]
            self.limit = [i for i in range(self.X.shape[1])]

        else:
            self.X = np.append(self.X, data_reading_model[0], axis=0)
            self.colours = np.append(self.colours, data_reading_model[1], axis=0)
            self.unique_label.append(data_reading_model[1][0])

            if self.type != data_reading_model[2]:
                self.type = GeneratorType.Mixture
                self.title = "Mixed Models"

            if shuffle:
                self.shuffle()
  

    def clearData(self):
        self.X, self.colours, self.y = None, None, None
        self.type = None
        self.title = ""
        self.limit = None
    

    def shuffle(self):
        p = np.random.permutation(self.colours.shape[0])
        self.X = self.X[p]
        if not self.y is None:
            self.y = self.y[p]
        self.colours = self.colours[p]



    def drawWhole(self, x = 0, y = 1):
        fig, ax = plt.subplots()

        for i, value in enumerate(self.unique_label):
            mask = np.where(self.colours == value, True, False)
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
    


    def drawResult(self, x = 0, y = 1, graph_type = "scatter"):
        fig, ax = plt.subplots()
        

        if graph_type == "scatter":

            if issubclass(type(self.model), SupervisedModel):
                for i, value in enumerate(self.unique_label):
                    mask = np.where(self.colours_test == value, True, False)
                    ax.scatter(x=self.X_test[mask, x],
                                y=self.X_test[mask, y],
                                color=ColourList[i],
                                marker='o', linewidths=0)
                self.model.makePlot(self.X_test, self.y_test, self.y_pred, x, y, ax)

            else:
                for i, value in enumerate(self.unique_label):
                    mask = np.where(self.colours == value, True, False)
                    ax.scatter(x=self.X[mask, x],
                                y=self.X[mask, y],
                                color=ColourList[i],
                                marker='o', linewidths=0)
                self.model.makePlot(self.X, x, y, self.X_transform, ax)
        

        elif graph_type == "matrix":
            
            if issubclass(type(self.model), SupervisedModel):

                # Prepare the values
                unique = np.unique(self.colours)
                legend_len = len(unique)
                label_dict = DataGenerator.getLabelDict(unique)

                con_matrix = np.zeros((legend_len, legend_len), dtype=int)
                for i in range(len(self.y_pred)):
                    con_matrix[label_dict[self.y_test[i]],
                               label_dict[self.y_pred[i]]] += 1

                # Make a confusion matrix based on the calculated value
                ax.imshow(con_matrix, cmap=LinearSegmentedColormap.from_list("", ["#FFFFFF", "#72E1D1"]))

                # Modify the chart
                ax.set_xticks(np.arange(legend_len), labels=unique)
                ax.set_yticks(np.arange(legend_len), labels=unique)
                ax.set_xlabel("Predicted values", fontsize=14)
                ax.set_ylabel("Actual values"   , fontsize=14)
                for pos in ['top', 'bottom', 'right', 'left']:
                    ax.spines[pos].set_edgecolor("#B5D8CC")
                
                for i in range(legend_len):
                    for j in range(legend_len):
                        ax.text(j, i, con_matrix[i, j], fontsize=16, ha="center", va="center")

        
        ax.tick_params(axis='x', colors="#8a7090")
        ax.tick_params(axis='y', colors="#8a7090")
        for pos in ['top', 'bottom', 'right', 'left']:
            ax.spines[pos].set_edgecolor("#B5D8CC")

        fig.suptitle("Linear data", fontsize=16, fontweight='bold')

        plt.show()
    

    @staticmethod
    def getLabelDict(unique):
        label_dict = {}
        id = 0
        for limit in unique:
            if not limit in label_dict.keys():
                label_dict[limit] = id
                id += 1
        return label_dict


