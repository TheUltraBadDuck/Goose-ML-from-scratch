import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from models.model import SupervisedModel, UnsupervisedModel



ColourList = ["#291D25", "#8A7090", "#823167", "#9D4BA2", "#FACFEC", "#D0BAFC"]
GlassColourList = ["#291D2544", "#8A709044", "#82316744", "#9D4BA244", "#FACFEC44", "#D0BAFC44"]



class DataBase:

    # Contructor
    def __init__(self):
        self.X = None
        self.y = None
        self.colours = None

        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        self.y_pred  = None
        self.X_transform = None
        self.model   = None
        self.limit   = []
    


    def x2Y(self, y):
        self.y = self.X[:, y]
        if y in self.limit:
            self.limit.remove(y)
    

    def colour2Y(self):
        self.y = np.copy(self.colours)
        unique = np.unique(self.y)
        id = 0
        for i in unique:
            if id != i:
                print("[Warning]: the label is missing {}. Assign the label colour continuously.".format(id))
            id += 1



    # Split data into training and testing
    def split(self, test_size=0.2, random_state=1234):
        if self.y is None:
            raise Exception("Non existent label. Use x2Y(y=...) or colour2Y() to assign the label.")
        
        self.X_train, self.X_test, \
            self.y_train, self.y_test,  \
            self.colours_train, self.colours_test = train_test_split(self.X, self.y, self.colours, test_size=test_size, random_state=random_state)


    # Get information of the data
    def info(self):
        print("General X: {}".format(self.X.shape))

        if not self.y is None:
            print("General y: {}".format(self.y.shape))

        if not self.X_train is None:
            print()
            print("Training X: {}".format(self.X_train.shape))
            print("Training y: {}".format(self.y_train.shape))
            print("Testing  X: {}".format(self.X_test.shape))
            print("Testing  y: {}".format(self.y_test.shape))

        if not self.y_pred is None:
            print("\nPredicted y: {}".format(self.y_pred))

        if not self.X_transform is None:
            print("\nTransformed groups: {}".format(self.X_transform))
        
        print("\nData information:")
        data = pd.DataFrame(self.X)
        data["y"] = self.y
        print(data.describe())

        unique, count = np.unique(self.colours, return_counts=True)
        print("\nNumber of colours: {}".format(unique))
        #if unique > 1:




    def drawWhole(self):
        pass


    def drawResult(self):
        pass



    def setModel(self, model):
        self.model = model

    

    def train(self):
        if self.model is None:
            raise Exception("The model has not been set yet. use setModel(...) to add a model.")
        if issubclass(type(self.model), SupervisedModel):
            self.model.fit(self.X_train[:, self.limit], self.y_train)
        else:
            self.model.fit(self.X)



    def test(self):
        if self.model is None:
            raise Exception("The model has not been set yet. use setModel(...) to add a model.")
        if issubclass(type(self.model), SupervisedModel):
            self.y_pred = self.model.predict(self.X_test[:, self.limit])
            return self.y_pred
        else:
            raise Exception("This object does not contain a supervised Model")



    def transform(self):
        if self.model is None:
            raise Exception("The model has not been set yet. use setModel(...) to add a model.")
        if issubclass(type(self.model), UnsupervisedModel):
            self.X_transform = self.model.transform(self.X)
            return self.X_transform
        else:
            raise Exception("This object does not contain a unsupervised Model")
    


    def setAndLearnModel(self, model, to_y=None, splitted_test_size=None):
        self.model = model
        if issubclass(type(self.model), SupervisedModel):

            if not to_y is None:
                if issubclass(type(to_y), int):
                    self.x2Y(to_y)

                elif (to_y == "c") or (to_y == "colour"):
                    self.colour2Y()
            
            if not splitted_test_size is None:
                self.split(splitted_test_size)

            self.model.fit(self.X_train[:, self.limit], self.y_train)
            self.y_pred = self.model.predict(self.X_test[:, self.limit])
            return self.y_pred
        
        else:
            self.model.fit(self.X[:, self.limit])
            self.X_transform = self.model.transform(self.X)
            return self.X_transform


