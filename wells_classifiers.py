# -*- coding: utf-8 -*-
"""
Created on Sun May  1 12:07:20 2016

@author: Michael
"""
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import svm
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

#Suppress FutureWarning
def fxn():
    warnings.warn("future", FutureWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
def accuracy_score(X,y,clf):
    scr = 0    
    y_pred = clf.predict(X)
    for i in range(0,len(y)):
        if(y_pred[i]==y.iloc[i]):
            scr+=1
    acc = float(scr)/len(y)  
    return acc

#Performs cross-validation on clf, outputting each fold to screen
def cross_val(X,y,clf,folds=5):
    cv_scores = model_selection.cross_val_score(clf,X,y,cv=folds)
    print("Cross-Validation with %i folds. Mean accuracy: %.4f" % (folds,np.mean(cv_scores)))
    
#Creates a training and test, then runs logisticRegression on each single feature, 
#returning training and test accuracy for each feature
def single_feat(X_train,X_test,y_train,y_test,clf):
    acc_df = pd.DataFrame(np.zeros((X_train.shape[1],2)),columns=['train','test'])
    for i in range(0,X_train.shape[1]):
        clf.fit(X_train[:,i:i+1],y_train)
        acc_df.iloc[i][0] = accuracy_score(X_train[:,i:i+1],y_train,clf)
        acc_df.iloc[i][1] = accuracy_score(X_test[:,i:i+1],y_test,clf)
    print(acc_df)
    return acc_df

# Scale features
def scale_X(X_train,X_test):
    scaler = StandardScaler()
    scaler.fit(X_train) 
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

# GridSearch Parameters
def grid_optimize(clf,param_grid,X,y):
    cv = cross_validation.StratifiedShuffleSplit(y, n_iter=5, test_size=0.2)
    grid = GridSearchCV(clf,param_grid,cv=cv)
    grid.fit(X,y)
    print("The best parameters are %s with a score of %0.2f" % (grid.best_params_, grid.best_score_))
    return grid

# Logistic Regression
def LR_mhs(X_train,X_test,y_train,y_test):
    LR_clf = linear_model.LogisticRegression()  
    LR_param_grid = {'penalty':['l1','l2']}
    LR_grid = grid_optimize(LR_clf,LR_param_grid,X_train,y_train)
    LR_clf = linear_model.LogisticRegression(penalty=LR_grid.best_params_['penalty'])
    LR_clf.fit(X_train,y_train)
#    LogReg_acc = single_feat(X_train,X_test,y_train,y_test,LR_clf)
    train_acc = accuracy_score(X_train,y_train,LR_clf)
    test_acc = accuracy_score(X_test,y_test,LR_clf)
    print("LR Classifier")    
    print("Training Accuracy: %.4f" % train_acc)
    print("Testing Accuracy: %.4f" % test_acc)
    return train_acc, test_acc, LR_clf
    
# SGDClassifier
def SGD_mhs(X_train_scaled,X_test_scaled,y_train,y_test):
    SGD_clf = linear_model.SGDClassifier(max_iter=1000,tol=1e-3)
    SGD_param_grid = {'penalty':['l1','l2','elasticnet'],'loss':['hinge','log','squared_loss'],'alpha':[.000003,.00001,.00003,.0001,.0003,.001]}
    SGD_grid = grid_optimize(SGD_clf,SGD_param_grid,X_train_scaled,y_train)
    SGD_clf = linear_model.SGDClassifier(loss=SGD_grid.best_params_['loss'],penalty=SGD_grid.best_params_['penalty'],alpha=SGD_grid.best_params_['alpha'],max_iter=1000,tol=1e-3)
#    SGD_acc = single_feat(X_train_scaled,X_test_scaled,y_train,y_test,SGD_clf)
    SGD_clf.fit(X_train_scaled,y_train)
    train_acc = accuracy_score(X_train_scaled,y_train,SGD_clf)
    test_acc = accuracy_score(X_test_scaled,y_test,SGD_clf)
    print("SGD Classifier")    
    print("Training Accuracy: %.4f" % train_acc)
    print("Testing Accuracy: %.4f" % test_acc)
    return train_acc, test_acc, SGD_clf

# SVM with rbf kernel Classifier
def SVM_mhs(X_train_scaled,X_test_scaled,y_train,y_test):
    SVM_clf = svm.SVC(kernel='rbf')
    C_range = [.01,300,10000000]
    gamma_range = [.000000001,.0003,1000]
   # SVM_param_grid = dict(gamma=gamma_range,C=C_range)
   # SVM_grid = grid_optimize(SVM_clf,SVM_param_grid,X_train_scaled,y_train)
   # SVM_clf = svm.SVC(gamma=SVM_grid.best_params_['gamma'],C=SVM_grid.best_params_['C'])
    SVM_clf = svm.SVC(gamma=.0003,C=100) 
    SVM_clf.fit(X_train_scaled,y_train)
    train_acc = accuracy_score(X_train_scaled,y_train,SVM_clf)
    test_acc = accuracy_score(X_test_scaled,y_test,SVM_clf)
    print("SVC Classifier with rbf kernel")    
    print("Training Accuracy: %.4f" % train_acc)
    print("Testing  Accuracy: %.4f" % test_acc)
    return train_acc, test_acc, SVM_clf

def RF_mhs(X_train,X_test,y_train,y_test):
    RF_clf = RandomForestClassifier(n_estimators = 10, max_depth = 5)  
    RF_param_grid = {'max_features':[None,'log2','sqrt']}
    RF_grid = grid_optimize(RF_clf,RF_param_grid,X_train,y_train)
    RF_clf = RandomForestClassifier(n_estimators=10,max_depth=5,max_features=RF_grid.best_params_['max_features'])
    RF_clf.fit(X_train,y_train)
    train_acc = accuracy_score(X_train,y_train,RF_clf)
    test_acc = accuracy_score(X_test,y_test,RF_clf)
    print("RF Classifier")    
    print("Training Accuracy: %.4f" % train_acc)
    print("Testing Accuracy: %.4f" % test_acc)
    return train_acc, test_acc, RF_clf    

# MAIN 
#Create training and test sets, scaled and not
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,y,test_size=0.35)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
scal=[]
for i in range(0,X.shape[1]):
    if max(X.iloc[:,i])!=1:
        scal.append(i)  
X_train_scaled.iloc[:,scal], X_test_scaled.iloc[:,scal] = scale_X(X_train.iloc[:,scal],X_test.iloc[:,scal])
del(scal)


algo_df = pd.DataFrame(np.zeros((4,2)),columns=['train','test'],index=['Logistic','SGD','SVM','RF'])

algo_df.iloc[1][0], algo_df.iloc[1][1], SGD_clf = SGD_mhs(X_train_scaled,X_test_scaled,y_train,y_test)
algo_df.iloc[0][0], algo_df.iloc[0][1], LR_clf = LR_mhs(X_train,X_test,y_train,y_test)
algo_df.iloc[2][0], algo_df.iloc[2][1], SVM_clf = SVM_mhs(X_train_scaled,X_test_scaled,y_train,y_test)
algo_df.iloc[3][0], algo_df.iloc[3][1], RF_clf = RF_mhs(X_train,X_test,y_train,y_test)
print(algo_df)


   
    