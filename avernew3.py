# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:33:52 2018

@author: Administrator
"""

import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error
from sklearn.cross_validation import cross_val_score
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_selection import SelectKBest, chi2  
from sklearn.pipeline import Pipeline  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.naive_bayes import BernoulliNB, MultinomialNB  
from sklearn.linear_model import RidgeClassifier  
from sklearn.linear_model import Perceptron  
from sklearn.neighbors import NearestCentroid  
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostRegressor,RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier  
from sklearn import metrics  
from time import time
from sklearn import tree
from collections import defaultdict
import pickle as p
import pandas as pd
from sklearn import linear_model
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression,Ridge,PassiveAggressiveRegressor,LassoLars
from sklearn import cross_validation,metrics
from sklearn import svm
from sklearn import linear_model
import numpy as np
from sklearn.linear_model import BayesianRidge, LinearRegression
from collections import defaultdict
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_predict
from xgboost import XGBRegressor,XGBClassifier
import xgboost as xgb
from sklearn import svm
import scipy
import time
import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from dateutil.parser import parse
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_squared_error,accuracy_score,roc_auc_score,f1_score,log_loss,mean_squared_error
from sklearn.cross_validation import cross_val_score
from sklearn.feature_selection import SelectFromModel
from tqdm import tqdm
from sklearn import linear_model
import os
import cv2
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
import random
#test = pd.read_csv(data_path+'d_test_A_20180102.csv',encoding='gb2312')
#k=['a20','a21','a22','a23','a24','date']
#k=['a5','a6','a7','a8','a9','a10','a11','a12','a13','a14','a15','a16','a17','a18','a19','a20','a21','a22','a23','a24','date','spp','spn']
#k=[]
#for i in k[:]:
 #   train.pop(i)
  #  test.pop(i)
#train=train.loc[train['day']>39]
#train=train.loc[train['y']<30]

test1 = pd.read_csv("../result/nasx.csv",header=None)
test2 = pd.read_csv("../result/{}_0331bx.csv",header=None)
test3 = pd.read_csv("../result/nas2.csv",header=None)
classes = ['collar_design_labels', 'neckline_design_labels', 'skirt_leng th_labels', 
           'sleeve_length_labels', 'neck_design_labels', 'coat_length_labels', 'lapel_design_labels', 
           'pant_length_labels']
len(classes)

test1.columns = ['image_id', 'class', 'label']
test2.columns = ['image_id', 'class', 'label']
test3.columns = ['image_id', 'class', 'label']
#print('{0}: {1}'.format(cur_class, len(df_load)))
l=list()
for i in range(test1.shape[0]):
    a1=test1['label'].values[i].split(';')
    a2=test2['label'].values[i].split(';')
    a3=test3['label'].values[i].split(';')
    a=list()
    for j in range(len(a1)):
        a.append((float(a1[j])+float(a2[j])+float(a3[j]))/3)
    tmp_list = a
    tmp_result = ''
    for tmp_ret in tmp_list:
        tmp_result += '{:.4f};'.format(tmp_ret)
        
    l.append(tmp_result[:-1])
    #l.append(a)
z=test1.copy()
z['label']=pd.DataFrame({'i':l})
z.to_csv('../result/av3x.csv', header=None, index=False) 


































