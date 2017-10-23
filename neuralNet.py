import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.layers.noise import GaussianNoise
from keras.utils import np_utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


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


def feature_extraction(inputFile, featureDict, text, label, num_features=125):
    """Generates letter features from text using TF-IDF, after dropping empty rows. Further reduces the dimensionality
    using Principle Component Analysis, with a default of 50 dimensions."""
    df = pd.read_csv(inputFile, encoding="utf8")
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

    return x, y, feature_list

X_test, Id, feature_dict = feature_extraction_test("test_set_x.csv", "Text", "Id")
X, y, feature_list = feature_extraction("cleaned_data.csv", feature_dict, 'Text', 'label')

"""This section is used if you do not want to make a prediction about the test set, but about the train/test split
Commment out the redefinition of y and uncomment the rest, proceed to line 94"""
y = np_utils.to_categorical(y, 5)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=51)
# y_train = np_utils.to_categorical(y_train, 5)
# y_test = np_utils.to_categorical(y_test, 5)


model = Sequential()
noise = GaussianNoise(0.5)
model.add(Dense(len(feature_list), init="uniform", input_dim=len(feature_list), activation="relu"))
'''This next line can be commented out to remove the Gaussian noise, to get results similar to our first NN submisison'''
model.add(noise)
model.add(Dense(100, init="uniform", activation="relu"))
model.add(Dense(50, init="uniform", activation="relu"))
model.add(Dense(25, init="uniform", activation="relu"))
model.add(Dense(5))
model.add(Activation("softmax"))
sgd = SGD(lr=0.01)
model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=["accuracy"])


"""Reverse the commenting below to evaluate train/test split"""
model.fit(X, y, nb_epoch=100, batch_size=32)
prediction = model.predict_classes(X_test)
# model.fit(X_train, y_train, nb_epoch=100, batch_size=128)
# (loss, accuracy) = model.evaluate(X_test, y_test, verbose=1)
# print("Train/test accuracy: {}".format(accuracy))

"""This is was used to get a very rough estimate of how well a classifier might perform in the competition
based on how well it lined up with our established top classifiers"""
# nn = np.genfromtxt('NNresults.csv', delimiter=",")
# lr = np.genfromtxt('LRresults5.csv', delimiter=",")
# NNprediction = nn[1:, 1]
# LRprediction = lr[1:, 1]
# print('\nAccuracy: {0}'.format(sum((prediction == NNprediction)).astype(float)/len(prediction)))
# print('\nAccuracy: {0}'.format(sum((prediction == LRprediction)).astype(float)/len(prediction)))

toCSV = np.array([Id, prediction]).T
np.savetxt("NNresults.csv", toCSV, delimiter=",")
