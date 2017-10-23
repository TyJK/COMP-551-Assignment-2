import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pickle
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from mpl_toolkits.mplot3d import Axes3D



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


def feature_extraction(inputFile, featureDict, text, label, num_features=120):
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
    """Uncomment this if an unseen dataset is being used with """
    #x = pca.fit_transform(x.toarray())
    x = x.toarray()
    feature_list = tf.get_feature_names()
    print("training features count: {}".format(len(feature_list)))
    print("fitted_sample_shape: {}".format(np.shape(x)))
    print("X shape ", x.shape)
    y = df[label]

    return x, y, feature_list


def polynomial_initialization(inputArray, polynomial=1):
    """A function not used in the text classification but capable of creating non-linear decision boundaries"""
    x = np.c_[inputArray[:, 1:]]
    polyX = np.empty(len(x))

    for i in range(0, polynomial):
        column = x.T**polynomial
        polyX = np.vstack([polyX, column])
        polynomial -= 1

    polyX = polyX.transpose()
    X = polyX[:, 1:]
    y = inputArray[:, 0]

    return X, y


def sigmoid(inputvalue):
    """The sigmoid function"""
    return 1 / (1 + np.exp(-inputvalue))


def log_likelihood(features, label, weightvector):
    """The calculation of minimized log liklihood, based on class slides"""
    weightedfeatures = np.dot(features, weightvector)
    ll = -np.sum(label * np.log(sigmoid(weightedfeatures)) + (1 - label) * np.log(1 - sigmoid(weightedfeatures)))

    return ll

def ridge_regularization(features, labels, lamb):
    """The programatic implementation of ridge regression, split between the first and second term of the equation"""
    first = np.dot(features.T, features,)
    first += lamb*np.eye(features.shape[0])
    first = np.linalg.inv(first)
    second = np.dot(features.T, labels)
    ridge = np.dot(first, second)

    return ridge

def logistic_regression(features, label, alpha, epsilon=0.001):
    """Logistic regression function that adds a bias term to the feature matrix before randomly initializing the weights.
    Count is used to control printing frequency. Change is the change in log likelihood, which terminates the loop once
     the change is smaller than epsilon (minimum 10000 iterations)"""
    bias = np.ones((features.shape[0], 1))
    Xbias = np.hstack((bias, features))
    weights = np.random.rand(Xbias.shape[1])
    count = 0
    change = 1
    ll = log_likelihood(Xbias, label, weights)
    while change > epsilon or count < 10000:
        weightedData = np.dot(Xbias, weights.T)
        #regularization = ridge_regularization(Xbias, label, 1)
        weights += alpha*np.dot(label-sigmoid(weightedData), Xbias) #+ regularization
        if count % 1000 == 0:
            oldLog = ll
            ll = log_likelihood(Xbias, label, weights)
            change = abs(oldLog - ll)
            print(ll)
        count += 1

    return weights


def one_vs_all(features, labels, alpha, epsilon=0.01):
    """One vs All implementation around the logistic regression function. Creates a list of unique labels. For each
    unique label, an empty label vector is initialized. If the label in the original label vector is equal to the
    current label, it becomes 0 in the new vector, else it becomes 1. Logistic regression is run n times, where n is
    the number of distinct labels. The weight list is an m by n matrix, where m is the number of labels and n the
    number of features. In essence for each example, n different classifiers are created."""
    distinctLabels = set(list(labels))
    weightList = []
    for i in distinctLabels:
        newLabels = []
        for label in labels:
            if label == i:
                newLabels.append(1)
            else:
                newLabels.append(0)
        weightList.append(logistic_regression(features, np.array(newLabels), alpha, epsilon))
        print("Progress: {} of {}".format(i+1, len(distinctLabels)))
    return weightList


def weight_data_matrix(X, weights):
    """From one vs all, the resulting dot product of the weights and features + bias is computed and returned as a
    generator"""
    weight_data = []
    dataplusbias = np.hstack((np.ones((X.shape[0], 1)), X))
    for i in weights:
        weight_data.append(np.dot(dataplusbias, i))

    yield weight_data


def activations(weighteddata):
    """The weight data generator is run through the sigmoid function. This was kept seperate to avoid memory issues that
    arose from having everything in one loop. It also allows as much memory to be saved as possible by utilizing
    generators as inputs when possible."""
    activation = []
    for j in weighteddata:
        activation.append(sigmoid(np.array(j)))
    return activation


def test(weightList, label=[]):
    """The final part of One vs All, the activated weights are transposed to become a n by m matrix, and for each row,
    the max value (the most confident classifier prediction is added to the prediction vector as the original class
    label. If a ground-truth label is included, the accuracy can be tested by comparing the prediction vector and the
    label vector and dividing the number of matching labels by the total number of predictions."""
    weightList = np.array(weightList)
    prediction = []

    for h in weightList.T:
        prediction.append(list(h).index(max(h)))

    if len(label) > 0:
        print('Accuracy: {0}'.format(sum((prediction == label)).astype(float)/len(prediction)))

    return prediction


def plot(features, labels, title):

    fig = plt.figure()
    ax = Axes3D(fig, elev=-150, azim=110)
    X_reduced = PCA(n_components=3).fit_transform(features)
    ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels,
               cmap=ListedColormap(['red', 'blue', 'yellow', 'green', 'purple'], name='Custom'), edgecolor='k', s=40)
    ax.set_title(title)
    ax.set_xlabel("1st eigenvector")
    ax.w_xaxis.set_ticklabels([])
    ax.set_ylabel("2nd eigenvector")
    ax.w_yaxis.set_ticklabels([])
    ax.set_zlabel("3rd eigenvector")
    ax.w_zaxis.set_ticklabels([])
    slavic = mpatches.Patch(color='red', label='Slavic')
    french = mpatches.Patch(color='blue', label='French')
    spanish = mpatches.Patch(color='yellow', label='Spanish')
    german = mpatches.Patch(color='green', label='German')
    polish = mpatches.Patch(color='purple', label='Polish')
    plt.legend(handles=[slavic, french, spanish, german, polish])

    plt.show()



X, id, feature_dict = feature_extraction_test("test_set_x.csv", "Text", "Id")
X_train, y_train, feature_list = feature_extraction("cleaned_data.csv", feature_dict, "Text", "label")

"""Reverse commenting to generate new weights rather than load existing ones"""
# weights = one_vs_all(X_train, y_train, 0.00001, 1)
# print(weights)
# pickle.dump(weights, open("LRweights2.pkl", 'wb'))
weights = pickle.load(open("LRweights.pkl", 'rb'))


testdatamatrix = weight_data_matrix(X, weights)
testactivation = np.array(activations(testdatamatrix))
test_prediction = test(testactivation)
toCSV = np.array([id, test_prediction]).T
np.savetxt("LRresults2.csv", toCSV, delimiter=",")

"""Uncomment to generate plots"""
# NNPredictions = np.genfromtxt('NNresults.csv', delimiter=",")
# LRPredictions = np.genfromtxt('LRresults5.csv', delimiter=",")
# plot(X_train, y_train, "PCA Reduced Training Data")
# plot(X, NNPredictions[1:, 1], "PCA Reduced Test Data with Logistic Regression Assigned Labels")
# plot(X, LRPredictions[1:, 1], "PCA Reduced Test Data\nLogistic Regression Assigned Labels")