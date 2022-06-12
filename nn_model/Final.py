#%%
import pickle
import pandas as pd
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout


def create_model3():
    model = Sequential()
    model.add(Dense(66, input_shape = (12,), activation = 'relu'))
    model.add(Dense(54, activation = 'elu'))
    model.add(Dense(32, activation = 'elu'))
    model.add(Dense(10, activation = 'elu'))
    model.add(Dense(5, activation = "softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer='RMSProp',metrics="accuracy")
    return model

class pred_cat():

    def __init__(self, path = os.getcwd() + "/nn_model.pkl"):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
    
    def change_type(self, data):
        if type(data) is list:
            X = pd.DataFrame(data).T
        else:
            X = data
        return X
    
    def extract_var(self, data):
        feat = ['sex', 'age', 'hasOtherComAccount', 'incomeYear', 'totalWealth', 'expInvestment', 'srcCapital', 'quotaCredit', 'category_rule_base', 'credit_info', 'lc_var', 'category_insurance']
        X = data[feat]
        return X

    def get_category(self, X):
        model = self.model
        X = self.change_type(X)
        X = self.extract_var(X)

        pred_cat = model.predict(X)
        res = pred_cat.tolist()
        for i in range(len(res)):
            if res[i] == 0:
                res[i] = '0-10萬'
            elif res[i] ==1:
                res[i] = '10-30萬'
            elif res[i] ==2:
                res[i] = '30-50萬'
            elif res[i] ==3:
                res[i] = '50-100萬'
            else:
                pass

        return res


#%%
#取資料測試
import os
path = os.getcwd()+ "/data/data_Y2.csv"
data = data = pd.read_csv(path)

classifier = pred_cat()
cat = classifier.get_category(ex1)
print("職稱: {} \n分類: {}".format(ex1, cat))



# %%
