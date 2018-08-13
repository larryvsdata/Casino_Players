# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 11:37:05 2018

@author: Erman
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from pandas import ExcelWriter
from pandas import ExcelFile
 
class PI():
    
    
    def __init__(self):
        self.df = pd.read_excel('PlayerData.xlsx', sheetname='datadump')
        self.X=None
        self.y=[]
        self.n_estimators=200
        self.clf=None
        
        self.splitRatio=0.33
        self.trainX=[]
        self.trainY=[]
        self.testX=[]
        self.testY=[]
        self.validationAccuracies=[]
        self.kFold=5
        self.results=None
        self.models=[]
       
        self.finalAccuracy=0
        
    def fillAndIndex(self):
        self.df.replace('null',0,inplace=True)
        self.df=self.df.reset_index(drop=True)
        
    def getElapsedTimes(self):
        
        timeList=list(self.df['install date'])
        latestTime=max(timeList)
        deltaTimes = list(map(lambda x: latestTime-x , timeList))
        elapsedDays = list(map(lambda x: x.days , deltaTimes))
        self.df['ElapsedDays']=elapsedDays
    
    def fixLevels(self):
        levels=self.df['current level'].values.tolist()
        levels = list(map(lambda x:  15 if x>15 else x, levels))
        self.df['current level']=levels
        
    def normalizeColumns(self):
        columnList=['coin balance','No. of spins','Total coin wins','Total coin bets','ElapsedDays']
        
        for column in columnList:
            listToBeNormalized=list(self.df[column])
            mu=np.mean(listToBeNormalized)
            sigma=np.std(listToBeNormalized)
            listToBeNormalized=list(map(lambda x: (x-mu)/sigma , listToBeNormalized))
            self.df[column]=listToBeNormalized
            
    def getXY(self):
        self.y=self.df['current level'].values.tolist()
        self.X=self.df.drop(['current level','playerid','install date'], axis=1).values.tolist()
    
    def trainTestSplit(self):
        self.trainX, self.testX,self.trainY, self.testY = train_test_split(self.X, self.y, test_size=self.splitRatio, random_state=42)
    
    def trainAndValidate(self):    
        validationRatio=1/float(self.kFold)
            
        for validation in range(self.kFold):
               print("Validation number : ", validation)
               clf=RandomForestClassifier(n_estimators=self.n_estimators)
                        
               trainX, self.validateX,trainY, self.validateY = train_test_split(self.trainX, self.trainY, test_size=validationRatio)
               clf.fit(trainX,trainY)
               
               outcome=clf.predict(self.validateX)
               self.validationAccuracies.append(accuracy_score(outcome,self.validateY))
               self.models.append(clf)
        
        
        self.clf=self.models[self.validationAccuracies.index(max(self.validationAccuracies))]
        del self.models[:]
        print("Validation Accuracies: ")
        print(self.validationAccuracies)
        
    def test(self):
            self.results=self.clf.predict( self.testX)
            self.finalAccuracy=accuracy_score(self.results,self.testY) 
        
    def predictAndScore(self):
#        self.results=self.model.predict(self.testX)
        print("Accuracy Score: ", accuracy_score(self.results,self.testY ))
        print("Confusion Matrix: ")
        print( confusion_matrix(self.results,self.testY ))

    
        
    def printResults(self):
       
       for ii in range(len(self.results)):
           print(self.testY[ii],self.results[ii]) 
           
    def plot_coefficients(self):
        coef = self.clf.feature_importances_
 
         # create plot
        importances = pd.DataFrame({'feature':self.df.drop(['current level','playerid','install date'], axis=1).columns.values,'importance':np.round(coef,3)})
        importances = importances.sort_values('importance',ascending=True).set_index('feature')
        print( importances)
        importances.plot.barh() 
           
    
        
        
        
if __name__ == '__main__':
    
    myPI=PI()
    myPI.fillAndIndex()
    myPI.getElapsedTimes()
    myPI.fixLevels()
    myPI.normalizeColumns()
    myPI.getXY()
    myPI.trainTestSplit()
    myPI.trainAndValidate()
    myPI.test()
    myPI.predictAndScore()
    myPI.plot_coefficients()