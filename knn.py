# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal
"""

from nltk.metrics.scores import accuracy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.decomposition import PCA
import operator
import numpy as np
import pandas as pd

alphabet=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
folds=4

"""Splits the data into training and validation sets"""
def spltDataset(X,Y,index):
    
    testSize=1.0/folds
    X_train, X_test, Y_train, Y_test, train_index, test_index =train_test_split(X,Y,index,test_size=testSize)
    
    return X_train, Y_train, train_index, X_test, Y_test, test_index 

def feature_extraction(inputFile, featureDict, text, label, num_features=125):
    """Generates letter features from text using TF-IDF, after dropping empty rows. Further reduces the dimensionality
    using Principle Component Analysis, with a default of 50 dimensions."""
    df = pd.read_csv("cleaned_data.csv", encoding="utf8",index_col=0)
    df[text].replace(np.nan, '', inplace=True)
    for idx, line in df.iterrows():
        try:
            words = line[text]
            newWords = ''.join(words.split()).lower()
            df.set_value(idx, text, newWords)
        except:
            pass
    tf = TfidfVectorizer(analyzer='char', encoding="utf8", vocabulary=featureDict, min_df=10)
    print("sample size: {}".format(len(df[text])))
    pca = PCA(n_components=num_features)
    x = tf.fit_transform(df[text])
    x = x.toarray()
    """Can be used when initializing model if some of the test data is unseen, to ensure feature alignment"""
    #x = pca.fit_transform(x.toarray())
    feature_list = tf.get_feature_names()
    print("training features count: {}".format(len(feature_list)))
    print("fitted_sample_shape: {}".format(np.shape(x)))
    print("X shape ", x.shape)
    y = df[label]
    ids = df.index
    return x, y, feature_list,ids


"""Gets the feature vectores of the Test Set"""
def feature_extraction_test(inputFile, text, label):
    df = pd.read_csv(inputFile, encoding="utf8")
    for idx, line in df.iterrows():
        try:
            words = line[text]
            newWords = ''.join(words.split())
            df.set_value(idx, text, newWords)
        except:
            pass
    df[text].replace(np.nan, '', inplace=True)
    df.to_csv("test_set_x_dropped_0.csv", encoding="utf8")
    tf = TfidfVectorizer(analyzer='char', encoding="utf8", min_df=10)
    x = tf.fit_transform(df[text])

    test_feature_list = tf.get_feature_names()
    indiceList = list(range(len(test_feature_list)))

    print("indiceList: {}".format(indiceList))

    test_feature_dict = {x[0]: x[1] for x in zip(test_feature_list, indiceList)}
    print("test_feature_list: {}".format(test_feature_dict))

    x = x.toarray()
    id = df[label]

    return x, id, test_feature_dict

"""Gets the distances between a sample and a set of data points"""
def calculateDistances(sample,points):
    return euclidean_distances(points, sample)

"""Gets the sorted indexes of an array in increasing order"""
def sortDistances(distances):
    return distances.argsort(axis=0)

"""Gets the k first elements of an array"""
def getClosest(k,arr):
    neighbors=[]
    x=k
    
    if k>len(arr):
        x=len(arr)
    
    for i in range(0,x):
        neighbors.append(arr[i][0])
        
    return neighbors

"""Gets the chosen class my taking the neighbors classes as votes """
def voteClass(neighbors, labels):
    votes={}
    for neighbor in neighbors:
        label=labels[neighbor]
        votes[label]=votes.get(label,0)+1
    return max(votes.items(), key=operator.itemgetter(1))[0]

"""KNN algorithm"""
def classify (trainVec, labels, testVec, k, ids,file):
    """Opens the file to write the results"""
    f=open(file,'w')
    f.write('Id,Category\n')
    result=[]
    
    """iterates through all the samples"""
    for i in range(0,testVec.shape[0]):
        sample=testVec[i].reshape(1,-1)
        """Calculates the distance tyo  data points"""
        distances=calculateDistances(sample,trainVec)
        """Sorts by distance"""
        sort=sortDistances(distances)
        """Gets the k closest neighbor's indexes"""
        neighbors=getClosest(k,sort)
        """Gets the chosen class"""
        chosenClass=voteClass(neighbors, labels)
        
        """Adds the result"""
        result.append(chosenClass)
#        print(str(ids[i])+','+str(chosenClass))
        f.write(str(ids[i])+','+str(chosenClass)+'\n')
    return result


"""Gets the test set features and id's"""
X_test, Id, feature_dict = feature_extraction_test("test_set_x.csv", "Text", "Id")
"""Gets the training set features"""
X, y, feature_list,indexes = feature_extraction("cleaned_data.csv", feature_dict, 'text', 'label')


"""Splits the dataset if a validation set is to be used"""
trainX,trainY,train_index,validX,validY,test_index =spltDataset(X,y,indexes)


"""Classifies the validation dataset CAN BE COMMENTED"""
#classif=classify(trainX,trainY.values,validX[0:4],11,test_index.tolist()[0:4],'resultTest')
#print('Validation accuracy: '+str(accuracy(classif,validY[0:4].values)))
    

"""Classifies the test dataset CAN BE COMMENTED"""
classif=classify(X,y.values,X_test[0:5],11,Id.tolist()[0:5],'resultTest')