# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
from sklearn.svm import LinearSVC
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf


class Modelo():
    # Captura de los datos
    trainig = pd.read_csv('E:/Maestrado/Machine Learning/Machine Learning I/DataSet/MNIST/fashion-mnist_train.csv',
                              sep=';', header=None)
    test = pd.read_csv('E:/Maestrado/Machine Learning/Machine Learning I/DataSet/MNIST/fashion-mnist_test.csv',
                       sep=';', header=None)


    # Divido los datos del trining y test
    X_trainig = trainig.iloc[1:, 1:].values
    Y_trainig = trainig.iloc[1:, 0:1].values.reshape(-1, 1)

    Y_trainig = Y_trainig.astype('int')
    X_trainig = X_trainig.astype('int')

    X_test = test.iloc[1:, 1:].values
    Y_test = test.iloc[1:, 0:1].values.reshape(-1, 1)

    X_test = X_test.astype('int')
    Y_test = Y_test.astype('int')

    x_train, x_eval, y_train, y_eval = train_test_split(X_trainig, Y_trainig, test_size=0.3, random_state=100)


    def ArbolDecision(self):
        # Profundidad 3, numero minimo de muestras requeridas para estar en un nodo hoja = 5.
        classifier = tree.DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=10,
                                                     min_samples_leaf=5, splitter='best')
        #Ajustando el clasificador
        classifier = classifier.fit(self.x_train, self.y_train)

        #Predicción
        y_pred = classifier.predict(self.x_eval)

        #Calculando precisión del clasificador
        acc = accuracy_score(self.y_eval, y_pred)

        #clf_entropy = tree.DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)
        #clf_entropy.fit(X_trainig, Y_trainig)
        #y_pred_en = clf_entropy.predict(X_test)

        return acc

        #print("Prediction in decision tree", y_pred)
        #print("Precision de Decision Tree", acc)

######### Fin de decision tree ######## accuracy para prediction - 0.805 #########

 
    def Random(self):
        clf_RF = RandomForestClassifier(n_estimators=100, max_depth=100, random_state=100, criterion='entropy',
            max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=5, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_jobs=None,
            oob_score=False, verbose=0, warm_start=False)

        clf_RF = clf_RF.fit(self.x_train, self.y_train.ravel())
        predict_RF = clf_RF.predict(self.x_eval)
        acc_RF = accuracy_score (self.y_eval, predict_RF)
    # print("Clasificacion de Random Forest", clf_RF.feature_importances_)

        return acc_RF

###### Fin Random Forest ####### accuracy para prediction - 0.875 #########


    def Lineal (self):
        lda = LinearDiscriminantAnalysis()
        model = lda.fit(self.x_train, self.y_train.ravel())

        pred_lda = model.predict(self.x_eval)
        acc_trining_LDA = lda.score(self.x_train, self.y_train)
        acc_test_LDA = lda.score(self.x_eval, self.y_eval)

    #print("Matriz de confusion de LDA ", confusion_matrix(pred_lda, Y_test))

        return classification_report(self.y_eval, pred_lda, digits=3), acc_trining_LDA, acc_test_LDA

#####Fin de Linear Discriminat Analysis####### Resulto para acc_trining_LDA -0.832 y para acc_test_LDA - 0.821


    def NaiveBG(self):
        GausNB = GaussianNB()
        GausNB.fit(self.x_train, self.y_train.ravel())

        y_pred = GausNB.predict(self.x_eval)

        return accuracy_score(self.y_eval, y_pred)

###### FIN de Naive Bayes-Gaussian ######## accuracy para prediction - 0.593 #############


    def NaiveBNB(self):

        BernNB = BernoulliNB(binarize=True)
        BernNB.fit(self.x_train, self.y_train.ravel())

        y_expet = self.y_eval
        y_pred = BernNB.predict(self.x_eval)

        return accuracy_score(y_expet, y_pred)

    ###### FIN Naive Bayes-Bernouli ####### accuracy para prediction - 0.720 ########


    def NaiveMultinomial(self):

        MultiNB = MultinomialNB()
        MultiNB.fit(self.x_train, self.y_train.ravel())

        y_expet = self.y_eval
        y_pred = MultiNB.predict(self.x_eval)

        return accuracy_score(y_expet, y_pred)

######## FIN Naive Bayes-Multinomial ###### accuracy para prediction - 0.66 #######


    def KNeighbors(self):

        KN = KNeighborsClassifier(n_neighbors=4, p=1, weights="distance")
        KN.fit(self.x_train, self.y_train.ravel())

        y_predKN = KN.predict(self.x_eval)
        acc_trining_KN = KN.score(self.x_train, self.y_train)
        acc_test_KN = KN.score(self.x_eval, self.y_eval)

        #report = classification_report(self.y_eval, y_predKN)

        #print(confusion_matrix(self.y_eval, y_predKN, sample_weight=None))

        #plt.hist(y_predKN, bins=50, density=1)
        #plt.show()

        return acc_trining_KN, acc_test_KN

######## Fin de KNN ####### Resulto para acc_trining_KN - 1.0 y para acc_test_KN - 0.862 #####



if __name__ == '__main__':
    model = Modelo()

    #print("Precision de DT:", model.ArbolDecision())
    #print("Random Forest", model.Random())
    #print("Linear Discriminant", model.Lineal())
    #print("Naive Bayes-Gaussian:", model.NaiveBG())
    #print("Naive Bayes-Bernouli:", model.NaiveBNB())
    #print("Naive Bayes-Multinomial:", model.NaiveMultinomial())
    #print("KNeighbors ", model.KNeighbors())
