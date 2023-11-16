import pandas as pd
from sklearn.model_selection import train_test_split

from models.model import SupervisedModel, UnsupervisedModel



ColourList = ["#291D25", "#8A7090", "#823167", "#9D4BA2", "#FACFEC"]



class DataBase:

    # Contructor
    def __init__(self):
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        self.y_pred  = None
        self.X_transform = None
        self.model   = None
        self.limit   = []
    

    # Split data into training and testing
    def split(self, test_size=0.2, random_state=1234):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)


    # Get information of the data
    def info(self):
        print("General X: {}".format(self.X.shape))
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


    def drawWhole(self):
        pass


    def drawTrain(self):
        pass


    def drawTest(self):
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
    


    def setAndLearnModel(self, model):
        self.model = model
        if issubclass(type(self.model), SupervisedModel):
            self.model.fit(self.X_train[:, self.limit], self.y_train)
            self.y_pred = self.model.predict(self.X_test[:, self.limit])
            return self.y_pred
        else:
            self.model.fit(self.X[:, self.limit])
            self.X_transform = self.model.transform(self.X)
            return self.X_transform