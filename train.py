# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:32:43 2020

@author: zmh
"""
import warnings
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
import joblib

data=pd.read_csv("final-huesken.csv")
data2=pd.read_csv("final-vks.csv").iloc[:74]
data3=pd.read_csv("final-reynold.csv")
data4=pd.read_csv("final-taka.csv")
data5=pd.read_csv("final-harboth.csv")

start=1
bo=357
X=data.iloc[:,start:bo]
y=data.iloc[:,bo]
Xtest=data2.iloc[:,start:bo][:74]
ytest=data2.iloc[:,bo][:74]
Xtest2=data3.iloc[:,start:bo]
ytest2=data3.iloc[:,bo]
ytest2=(100-ytest2)/100
ytest=(100-ytest)/100
X41=data4.iloc[:,start:bo][:662]
y4=data4.iloc[:,bo][:662]
y4=y4/100
X51=data5.iloc[:,start:bo] 
y5=data5.iloc[:,bo]
y5=1-y5/100

thresh=0.9
def evaluate(ytrue,pred): 
    out=np.vstack((pred,ytrue.values)).T
    dat=pd.DataFrame(out)
    pcc=dat.corr().iloc[0,1]
    #score=0
    #truth = [1 if i>=thresh else 0 for i in ytrue]
    #fpr, tpr, _ = roc_curve(truth, pred)
    #roc_auc = auc(fpr, tpr)       
    return pcc
sorted_idx=[312, 59, 313, 58, 20, 328, 332, 0, 42, 38, 47, 261, 318, 317, 50, 323, 46, 334, 341, 342, 347, 349, 56, 316, 326, 331, 48, 39, 355, 57, 350, 322, 321, 320, 272, 145, 344, 54, 41, 271, 337, 339, 351, 49, 52, 40, 241, 13, 340, 314, 345, 346, 329, 354, 353, 338, 319, 44, 324, 259, 287, 74, 315, 63, 330, 348, 352, 325, 110, 327, 336, 335, 276, 53, 144, 267, 43, 153, 292, 55, 51, 333, 218, 297, 18, 28, 281, 235, 112, 296, 36, 300, 299, 288, 284, 302, 343, 279, 195, 190, 150, 146, 61, 214, 274, 117, 290, 258, 19, 286, 67, 245, 199, 21, 29, 303, 301, 158, 45, 77, 278, 250, 148, 91, 243, 283, 156, 186, 238, 174, 172, 187, 206, 310, 147, 159, 60, 70, 185, 160, 89, 305, 123, 252, 32, 240, 30, 102, 107, 275, 125, 27, 289, 213, 68, 92, 131, 282, 188, 189, 33, 191, 262, 285, 98, 154, 309, 212, 246, 244, 304, 223, 152, 203, 106, 209, 231, 224, 169, 149, 161, 119, 103, 109, 139, 157, 127, 205, 35, 251, 31, 220, 215, 211, 81, 204, 256, 307, 308, 196, 260, 66, 192, 25, 24, 73, 1, 22, 277, 78, 79, 248, 23, 85, 16, 15, 6, 128, 101, 64, 71, 72, 95, 76, 93, 3, 163, 194, 257, 143, 151, 253, 249, 234, 232, 164, 165, 229, 167, 221, 219, 170, 173, 175, 201, 198, 179, 197, 255, 193, 263, 266, 140, 293, 291, 273, 294, 265, 264, 311, 306, 62, 298, 217, 8, 10, 130, 202, 65, 216, 295, 7, 226, 14, 222, 225, 227, 230, 280, 270, 269, 239, 37, 184, 26, 254, 242, 177, 183, 80, 69, 104, 166, 141, 137, 87, 115, 155, 83, 178, 171, 181, 122, 75, 11, 34, 132, 100, 105, 124, 114, 12, 9, 135, 134, 120, 121, 247, 99, 228, 176, 237, 236, 162, 168, 208, 182, 90, 207, 86, 82, 180, 88, 200]
for i in range(334,335):
    X1=X[X.columns[sorted_idx][:i]]
    X2=Xtest[X.columns[sorted_idx][:i]]
    X3=Xtest2[X.columns[sorted_idx][:i]]
    X4=X41[X.columns[sorted_idx][:i]]  
    X5=X51[X.columns[sorted_idx][:i]]        
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")      
        lgb1= lgb.LGBMRegressor(num_iterations=79,    
                                min_data=20,
                                    max_depth=11,
                                    subsample=0.93,
                                    feature_fraction=0.147,
                                    subsample_freq=1,
                                    lambda_l1=0.28,  
                                    lambda_l2=1.9,
                                    learning_rate=0.1,
                                    n_jobs=-1,
                                    verbose=-1,
                                   )#Different versions of the same parameter may get different results, I don’t know why 
        lgb1.fit(   X1[:2182],
                    y[:2182],
                    verbose=False)           
        joblib.dump(lgb1, "lgbmodule.m") 
        #lgb1=joblib.load("lgbmodule.m")                      
        pred=lgb1.predict(X1[2182:])
        pred2=lgb1.predict(X2)
        pred3=lgb1.predict(X3)
        pred4=lgb1.predict(X4)
        pred5=lgb1.predict(X5)
        a=evaluate(y[2182:],pred)
        b=evaluate(ytest,pred2)
        c=evaluate(ytest2,pred3)
        d=evaluate(y4,pred4)  
        e=evaluate(y5,pred5)
        print('pcc:','%.4f' %a,'%.4f' %b,'%.4f' %c,'%.4f' %d,'%.4f' %(e))        