import pandas as pd
import numpy as np
from time import time
import winsound

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from langdetect import detect_langs
from sklearn import svm
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier


def feature_extraction(inputFile):

    df = pd.read_csv(inputFile, encoding="utf8")
    df['text'].replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    tf = TfidfVectorizer(analyzer='char', encoding="utf8", min_df=10)
    x = tf.fit_transform(df['text'])
    x = x.toarray()
    y = df['label']

    return x, y


def clean_data(inputFile, cutoff=0.95):
    """Drops all empty rows, and initializes a number of counter variables. Uses the langdetect library to generate a
    language code and confidence. This is then split into component parts. If the identifier is 'en' for english, and
    the confidence is above the cutoff (0.95 used to process data), the index of that row is added to a list. Else if
    the labels ISO code is not the same as the detected language and the confidence is above the cutoff, that index is
    also added to the list. A progress counter and timer were added for convenience as the cleaner took a long time to
    run. Once complete, all rows of the corresponding indices were dropped from the table. This dataframe was then
    saved to a csv. Relevant statistics are printed at time of termination."""
    ISOcodes = {'sk': 0, 'fr': 1, 'es': 2, 'de': 3, 'pl': 4}

    df = pd.read_csv(inputFile, encoding="utf8")
    df['text'].replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    total = len(df)
    englishCount, misclassifiedCount, count = 0, 0, 0
    hitList = []
    startTime = time()
    for line in df.iterrows():
        label = line[1]["label"]
        text = line[1]["text"]
        try:
            detectedLanguage = detect_langs(text)
            language = str(detectedLanguage[0]).split(":")
            if language[0] == 'en':
                if float(language[1]) > cutoff:
                    englishCount += 1
                    hitList.append(count)
            elif label != ISOcodes[language[0]]:
                if float(language[1]) > cutoff:
                    misclassifiedCount += 1
                    hitList.append(count)
        except:
            pass

        count += 1
        if count % 1000 == 0:
            percentComplete = count*100/total
            now = time()
            timeLeft = (1 - count/total)*((now-startTime)/60)/(count/total)
            timeLeft = str(round(timeLeft, 2)).split(".")
            minutes = timeLeft[0]
            seconds = (float(timeLeft[1])/100)*60
            print("Percent Complete: {}%".format(round(percentComplete, 2)))
            print("Time Left: {}:{:02d}".format(minutes, int(seconds)))
    df.drop(df.index[hitList], inplace=True)

    now = time()
    print("Number of English examples removed: {}".format(englishCount))
    print("Number of misclassified examples removed: {}".format(misclassifiedCount))
    print("Number of rows originally in dataframe: {}".format(total))
    print("Percent of training examples classified as English: {}%".format(round(englishCount*100/total, 2)))
    print("Percent of training examples classified as incorrect: {}%".format(round(misclassifiedCount*100/total, 2)))
    print("New dataframe length: {}".format(len(df)))
    print("Actual time taken in minutes: {}".format((now-startTime)/60))

    return df


def sk_test_suit(X, y):
    """A function to rapidly test a large number of sci-kit learn classifiers for the purposes of establishing baselines
    for the various datasets. After creating the dictionary, it is looped through and each classifier is executed and
    scored."""

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    classifierDict = {"Random Forest": RandomForestClassifier(),
                      "Logistic Regression": LogisticRegression(),
                      "Linear Discriminant Analysis": LinearDiscriminantAnalysis(),
                      "Gaussian Naive Bayes": GaussianNB(),
                      "Neural Network": MLPClassifier()}


    try:
        for k, v in classifierDict.items():
            clf = v.fit(X_train, y_train)
            training_score = cross_val_score(clf, X_train, y_train)
            testing_score = cross_val_score(clf, X_test, y_test)
            print(k)
            print('Sk-learn {0} training accuracy: {1}'.format(k, training_score.mean()))
            print('Sk-learn {0} testing accuracy: {1}'.format(k, testing_score.mean()))
    except:
        winsound.PlaySound('sound.wav', winsound.SND_FILENAME)



data = clean_data("training_set.csv", 0.85)
data.to_csv("cleaned_data.csv", encoding="utf8")
cleanX, cleanY = feature_extraction("cleaned_data.csv")
sk_test_suit(cleanX, cleanY)
