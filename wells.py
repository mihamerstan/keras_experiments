# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from datetime import datetime
#from sklearn.metrics import classification_report

#Creates a matrix binary feature-columns for each categorical value
def oneHot(vec):
    le = preprocessing.LabelEncoder()
    ohe = preprocessing.OneHotEncoder()
    labs = le.fit_transform(vec)
    mat_out = pd.DataFrame(ohe.fit_transform(labs.reshape(-1,1)).todense(),columns = le.classes_)
    return mat_out

# Returns accuracy of clf predictions of y based on X (correct pred / total)


#One hot the categoricals
def oneHotwell(features_df):
    basin_mat = oneHot(features_df['basin'])
    #publicmtg_mat = oneHot(features_df['public_meeting'])
    #permit_mat = oneHot(features_df['permit'])
    extraction_mat = oneHot(features_df['extraction_type_class'])
    management_mat = oneHot(features_df['management_group'])
    payment_mat = oneHot(features_df['payment_type'])
    quality_mat = oneHot(features_df['quality_group'])
    quantity_mat = oneHot(features_df['quantity_group'])
    source_mat = oneHot(features_df['source_class'])
    waterpoint_mat = oneHot(features_df['waterpoint_type_group'])
    
    return pd.concat([basin_mat, \
    extraction_mat, management_mat, payment_mat, quality_mat,\
    quantity_mat, source_mat, waterpoint_mat],axis=1)

#Import training data
labels_df=pd.read_csv("Training_set_labels.csv").reset_index(drop=False)
features_df=pd.read_csv("Training_set.csv").reset_index(drop=False)

#Turn T/F features into binary
permit = features_df['permit'].copy()
public_meeting = features_df['public_meeting'].copy()
permit[permit==True]=1
permit[permit!=True]=0
public_meeting[public_meeting==True]=1
public_meeting[public_meeting!=True]=0

#Create well age vector
date_recorded = pd.Series(np.zeros(len(features_df['construction_year'])))
age = pd.Series(np.zeros(len(features_df['construction_year'])),name='age')
const_yr = features_df['construction_year'].copy()
i = 0
for row in features_df['date_recorded']:
    date_recorded[i] = datetime.strptime(row,'%Y-%M-%d').year
    age[i]= date_recorded[i]-const_yr.iloc[i] 
    i +=1
age[age>1000]=0
# Create X Data Frame, y Series
usable_features = features_df[['amount_tsh', 'gps_height', 'num_private', 'population']].copy()
oneHotmat = oneHotwell(features_df)    
X = pd.concat([usable_features, age, permit, public_meeting, oneHotmat],axis=1)
y = labels_df['status_group'].copy()
del(date_recorded,const_yr,age,permit,public_meeting,oneHotmat,usable_features,row,i)