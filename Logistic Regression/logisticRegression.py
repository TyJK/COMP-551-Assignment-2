import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split



def feature_extraction(inputFile, num_features=50):
    """Generates letter features from text using TF-IDF, after dropping empty rows. Further reduces the dimensionality
    using Principle Component Analysis, with a default of 50 dimensions."""
    df = pd.read_csv(inputFile, encoding="utf8")
    df['text'].replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    tf = TfidfVectorizer(analyzer='char', encoding="utf8", min_df=10)
    pca = PCA(n_components=num_features)
    x = tf.fit_transform(df['text'])
    x = pca.fit_transform(x.toarray())
    y = df['label']

    return x, y


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
        weights += alpha*np.dot(label-sigmoid(np.dot(Xbias, weights.T)), Xbias)
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


def test(weightList, label=None):
    """The final part of One vs All, the activated weights are transposed to become a n by m matrix, and for each row,
    the max value (the most confident classifier prediction is added to the prediction vector as the original class
    label. If a ground-truth label is included, the accuracy can be tested by comparing the prediction vector and the
    label vector and dividing the number of matching labels by the total number of predictions."""
    weightList = np.array(weightList)
    prediction = []
    labels = list(y)
    for h in weightList.T:
        prediction.append(list(h).index(max(h)))

    if len(labels) > 0:
        print('Accuracy: {0}'.format(sum((prediction == label)).astype(float)/len(prediction)))

    return prediction


X, y = feature_extraction("cleaned_data.csv")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=51)


weights = one_vs_all(X_train, y_train, 0.0001, 0.05)
print(weights)
pickle.dump(weights, open("weights2.pkl", 'wb'))
weights = pickle.load(open("weights.pkl", 'rb'))



traindatamatrix = weight_data_matrix(X_train, weights)
trainactivation = np.array(activations(traindatamatrix))
training_prediction = test(trainactivation, y_train)
testdatamatrix = weight_data_matrix(X_test, weights)
testactivation = np.array(activations(testdatamatrix))
test_prediction = test(testactivation)
