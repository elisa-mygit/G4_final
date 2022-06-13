#%%
import pickle
import pandas as pd
import numpy as np
import os
import csv
from keras.models import Sequential
from keras.layers import Dense

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
    
    def get_lcvar(data):
        path = os.getcwd() + "/公開公司資訊.csv" #需改動資料位置
        stock = pd.read_csv(path)
        comp_name = list(stock.公司名稱)
        comp_abr = list(stock.公司名稱)
        comp = list(data.company)

        lc_var = []
        for c in comp:
            ans = 0
            try:
                for lc in comp_name:
                    if c in lc:
                        ans = 1
                    elif lc in c:
                        ans = 1
                    else: 
                        pass
            except:
                pass

            try:
                for lc in comp_abr:
                    if c in lc:
                        ans = 1
                    elif lc in c:
                        ans = 1
                    else: 
                        pass
            except:
                pass

        lc_var.append(ans)
        data["lc_var"] = lc_var
        return data   
    
    def get_creditinfo(self, data, credit_file):
        '''若徵信資料為另一檔案(credit_file)'''
        credit_file_list=[]
        with open(credit_file) as csvfile:
            csvDictReader = csv.DictReader(csvfile)
            for row in csvDictReader:
                credit_file_list.append(row['ssn'])
        
        data_ssn_list=[]

        for row in data:
            data_ssn_list.append(row['ssn'])

        credit_info=[]
        for i in range(0,len(data_ssn_list)):
            if data_ssn_list[i] in credit_file_list:
                credit_info.append('1')
            else:
                credit_info.append('0')
        
        data["credit_info"] = credit_info
    
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

        X = self.get_lcvar(X)
        try: 
            X =  self.get_creditinfo(X)
        except:
            pass    
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
cat = classifier.get_category(data)
