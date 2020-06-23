# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 09:31:48 2020

@author: zmh
"""
import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

def evaluate(ytrue,pred): 
    out=np.vstack((pred,ytrue.values)).T
    dat=pd.DataFrame(out)
    pcc=dat.corr().iloc[0,1]      
    return pcc

data=pd.read_csv("final-huesken.csv")
data2=pd.read_csv("final-vks.csv").iloc[:74]

start=1
bo=357
X=data.iloc[:,start:bo]
y=data.iloc[:,bo]
Xtest=data2.iloc[:,start:bo][:74]
ytest=data2.iloc[:,bo][:74]
 
#PCC特征排序
PCC=data.corr()
ch=[]
for i in range(1,357):
       ch.append(abs(PCC.iloc[357,i]))
sortid=sorted(range(len(ch)), key=lambda k: ch[k])
#lgb特征重要性排序

lgb1 = lgb.LGBMRegressor(n_estimators=100,max_depth=7)  
lgb1.fit(X[:2182],y[:2182])  
pred=lgb1.predict(X[2182:])
pred2=lgb1.predict(Xtest)
pred3=lgb1.predict(Xtest2)
pred4=lgb1.predict(X41)
pred5=lgb1.predict(X51)
a=evaluate(y[2182:],pred)
b=evaluate(ytest,pred2)

print('pcc:','%.4f' %a,'%.4f' %b)
model = SelectFromModel(lgb1, prefit=True)  
feature_importance = lgb1.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
feat=pd.DataFrame(X.columns,feature_importance)
sorted_idx= np.argsort(feature_importance)

pos = np.arange(sorted_idx.shape[0])
col=X.columns[sorted_idx]
plt.figure(figsize=(8,70))
plt.barh(pos, feature_importance[sorted_idx])
plt.yticks(pos, X.columns[sorted_idx],fontsize=10)
plt.ylabel('Features')
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show() 