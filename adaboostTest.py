import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
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
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

# https://machinelearningmastery.com/boosting-and-adaboost-for-machine-learning/
def generic_clf(X_train, Y_train, X_test, Y_test, clf):
    clf.fit(X_train,Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)

def plot_error_rate(er_train, er_test, lang, topIterCount):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,topIterCount,topIterCount/10))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('language{} Error rate vs number of iterations'.format(lang), fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')
    plt.show()

def plot_score_sk(scoreList, topIterCount):
    df_score = pd.DataFrame(scoreList, columns=['score'])
    plot1 = df_score.plot(linewidth = 3, figsize = (8,6),
            color = ['darkblue'], grid = True)
    plot1.set_xlabel('Number of iterations', fontsize = 12)
    plot1.set_xticklabels(range(0,topIterCount,topIterCount/10))
    plot1.set_ylabel('Score', fontsize = 12)
    plot1.set_title('Score vs number of iterations', fontsize = 16)
    plt.show()

def plot_error_rate_split(er_train, er_test, lang, topSplitPerc):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('Split Percentage', fontsize = 12)
    split_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    plot1.set_xticklabels(split_range)
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('language{} Error rate vs train/test split size'.format(lang), fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')
    plt.show()

def plot_error_rate_depth(er_train, er_test, lang, topTreeDepth):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth = 3, figsize = (8,6),
            color = ['lightblue', 'darkblue'], grid = True)
    plot1.set_xlabel('tree depth', fontsize = 12)
    plot1.set_xticklabels(range(1,topTreeDepth,1))
    plot1.set_ylabel('Error rate', fontsize = 12)
    plot1.set_title('language{} Error rate vs tree max depth size'.format(lang), fontsize = 16)
    plt.axhline(y=er_test[0], linewidth=1, color = 'red', ls = 'dashed')
    plt.show()

def train(X_train, Y_train, X_test, Y_test, n_estimators, base_estimator):
    # Each instance in the training dataset is weighted. The initial weight is set to:
    # weight(xi) = 1/n
    # Where xi is the ith training instance and n is the number of training instances.
    # m = num of features

    clsList = []
    clsWeightList = []

    m = np.shape(X_train)[0]
    m1 = np.shape(X_test)[0]
    # Initialize weights
    D = np.ones(m)/m
    z_norm = sum(D)
    pred_train = np.zeros(m)
    pred_test = np.zeros(m1)
    
    count_num = 0
    for i in range(n_estimators):
        # Fit a classifier with the specific weights
        base_estimator.fit(X_train, Y_train, sample_weight = D)
        pred_train_i = base_estimator.predict(X_train)
        pred_test_i = base_estimator.predict(X_test)
        # Indicator function
        pred_result_terror = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        pred_result_binary = [x if x == 1 else -1 for x in pred_result_terror]
        weighted_error = np.dot(D, pred_result_terror)
        # Alpha
        alpha_i = float(0.5 * np.log((1.0 - weighted_error) / float(weighted_error)))

        # New weights
        D = np.multiply(D, np.exp([float(x) * alpha_i for x in pred_result_binary]))
        # Normalization
        z_norm = sum(D)
        D = np.divide(D, z_norm)

        # Add to prediction
        pred_train = [sum(x) for x in zip(pred_train, [x * alpha_i for x in pred_train_i])]
        pred_test = [sum(x) for x in zip(pred_test, [x * alpha_i for x in pred_test_i])]

        count_num = count_num + 1
        percentComplete = count_num * 100 / n_estimators 
        print("Percent Complete: {}%".format(round(percentComplete, 2)))

        clsList.append(base_estimator)
        clsWeightList.append(alpha_i)
    
    pred_train = np.sign(pred_train)
    pred_test = np.sign(pred_test)

    dfPred = pd.DataFrame(pred_test, columns=['prediction'])
    dfytest = pd.DataFrame(Y_test, columns=['label'])
    # dfytest.to_csv("split_X_test_quick.csv", encoding="utf8")
    # dfPred.to_csv("split_X_test_predict_quick.csv", encoding="utf8")
    # Return error rate in train and test set 
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test), \
           clsList, clsWeightList

def training(X_train, Y_train, n_estimators, base_estimator):
    # Each instance in the training dataset is weighted. The initial weight is set to:
    # weight(xi) = 1/n
    # Where xi is the ith training instance and n is the number of training instances.
    # m = num of features
    clsList = []
    clsWeightList = []
    m = np.shape(X_train)[0]
    D = np.ones(m)/m
    pred_train = np.zeros(m)    
    count_num = 0
    for i in range(n_estimators):
        # Fit a classifier with the specific weights
        base_estimator.fit(X_train, Y_train, sample_weight = D)
        pred_train_i = base_estimator.predict(X_train)
        # Indicator function
        pred_result_terror = [int(x) for x in (pred_train_i != Y_train)]
        # Equivalent with 1/-1 to update weights
        pred_result_binary = [x if x == 1 else -1 for x in pred_result_terror]
        # The misclassification rate is calculated for the trained model. 
        # weighted_error = sum(w(i) * terror(i)) / sum(w)
        # w is the weight for training instance i and 
        # terror is the prediction error for training instance i 
        # which is 1 if misclassified and 0 if correctly classified.
        weighted_error = np.dot(D, pred_result_terror)#/ sum(D)
        # Alpha
        # alpha = float(0.5*log((1.0-error) / (error+1e-15)))
        alpha_i = float(0.5 * np.log((1.0 - weighted_error) / float(weighted_error)))
        # New weights
        D = np.multiply(D, np.exp([float(x) * alpha_i for x in pred_result_binary]))
        # Normalization
        z_norm = sum(D)
        D = np.divide(D, z_norm)
        # Add to prediction
        clsList.append(base_estimator)
        clsWeightList.append(alpha_i)
        count_num = count_num + 1
        percentComplete = count_num * 100 / n_estimators 
        print("Training Percent Complete: {}%".format(round(percentComplete, 2)))
    return clsList, clsWeightList

def feature_extraction(inputFile, featureDict, lang):
    df = pd.read_csv(inputFile, encoding="utf8")
    for idx, line in df.iterrows():
        try:
            words = line["text"]
            newWords = ''.join(words.split())
            df.set_value(idx, 'text', newWords)
            if line['label']== lang: # set it to 1, otherwise to -1
                df.set_value(idx, 'label', 1)
            else:
                df.set_value(idx, 'label', -1)
        except:
            pass

    y = df['label']
    df['text'].replace(np.nan, '', inplace=True)
    tf = TfidfVectorizer(analyzer='char', encoding="utf8", vocabulary=featureDict, min_df=10)
    x = tf.fit_transform(df['text'])
    feature_list = tf.get_feature_names()
    x = x.toarray()
    print(len(df.index))
    print(len(x))
    return x, y, feature_list

def feature_extraction_for_sklearn(inputFile, featureDict):
    df = pd.read_csv(inputFile, encoding="utf8")
    for idx, line in df.iterrows():
        try:
            words = line["text"]
            newWords = ''.join(words.split())
            df.set_value(idx, 'text', newWords)
        except:
            pass

    y = df['label']
    df['text'].replace(np.nan, '', inplace=True)
    tf = TfidfVectorizer(analyzer='char', encoding="utf8", vocabulary=featureDict, min_df=10)
    x = tf.fit_transform(df['text'])
    feature_list = tf.get_feature_names()
    x = x.toarray()
    return x, y, feature_list

def feature_extraction_test(inputFile):
    df = pd.read_csv(inputFile, encoding="utf8")
    for idx, line in df.iterrows():
        try:
            words = line['Text']
            newWords = ''.join(words.split())
            df.set_value(idx, 'Text', newWords)
        except:
            pass
    df['Text'].replace(np.nan, '', inplace=True)
    tf = TfidfVectorizer(analyzer='char', encoding="utf8", min_df=10)
    x = tf.fit_transform(df['Text'])

    test_feature_list = tf.get_feature_names()
    indiceList = list(range(len(test_feature_list)))
    test_feature_dict = {x[0]:x[1] for x in zip(test_feature_list, indiceList)}
    x = x.toarray()
    return x, test_feature_dict

def test(test_x_ds, classifiers, alphas, itr, lang):
    pred_test = np.zeros(np.shape(test_x_ds)[0])
    pred_test_weight = np.zeros(np.shape(test_x_ds)[0])
    count_i = 0
    for clsi, alpi in zip(classifiers, alphas):
        results = clsi.predict(test_x_ds)
        results_neg_pos_ones = [x if x == 1 else -1 for x in results]
        pred_test = [sum(x) for x in zip(pred_test, [float(x) * alpi for x in results_neg_pos_ones])]
        pred_test_weight = pred_test
        pred_test = np.sign(pred_test)
        count_i = count_i + 1
        percentComplete_test = count_i * 100 / len(classifiers)
        print("Testing Percent Complete: {}%".format(round(percentComplete_test, 2)))
    data = {'text_label': pred_test, 'weight': pred_test_weight}
    pred_test_df = pd.DataFrame(data)
    pred_test_df.to_csv("test_set_y_{}_{}.csv".format(lang, itr), encoding="utf8") 
    return pred_test_df

# Find best n_estimators for each language classifier
def itrValidationPhase(lang, maxRange, stepSize, featureDict, treeDept, split):
    cleanX0, cleanY0, trainFeatureList = feature_extraction("cleaned_data.csv", featureDict, lang)
    # Split train/test samples
    X_train, X_test, y_train, y_test = train_test_split(cleanX0, cleanY0, test_size=split, random_state=42)
    # Fit Adaboost classifier using a decision tree as base estimator
    er_tree = generic_clf(X_train, y_train, X_test, y_test, clf_tree)
    # Fit a simple decision tree first
    clf_tree = DecisionTreeClassifier(max_depth = treeDept, random_state = 42)
    # Test with different number of iterations
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    
    itr_range = range(1, maxRange, stepSize)

    for i in itr_range: 
        print("range: {}".format(i))
        errs = train(X_train, y_train, X_test, y_test, i, clf_tree)
        er_train.append(errs[0])
        er_test.append(errs[1])

    # Compare error rate vs number of iterations
    plot_error_rate(er_train, er_test, lang, maxRange)

def splitValidationPhase(lang, featureDict, treeDept, itr):
    cleanX0, cleanY0, trainFeatureList = feature_extraction("cleaned_data.csv", featureDict, lang)
    # Split train/test samples
    split_range = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    for i in split_range:
        X_train, X_test, y_train, y_test = train_test_split(cleanX0, cleanY0, test_size=i, random_state=42)
        clf_tree = DecisionTreeClassifier(max_depth = treeDept, random_state = 42)
        er_tree = generic_clf(X_train, y_train, X_test, y_test, clf_tree)
        er_train, er_test = [er_tree[0]], [er_tree[1]]
        print("splitPhase range: {}".format(i))

        print("splitValidationPhase, X_train size: {}", format(np.shape(X_train)))
        print("splitValidationPhase, X_test size: {}", format(np.shape(X_test)))

        errs = train(X_train, y_train, X_test, y_test, itr, clf_tree)
        er_train.append(errs[0])
        er_test.append(errs[1])
    pickle.dump(er_train, open("er_train_{}_{}_{}_Clean.pkl".format(lang, treeDept, itr), 'wb'))
    pickle.dump(er_test, open("er_test_{}_{}_{}_Clean.pkl".format(lang, treeDept, itr), 'wb'))
    plot_error_rate_split(er_train, er_test, lang, 0.5)

def treeDeptValidationPhase(lang, featureDict, split, itr):
    cleanX0, cleanY0, trainFeatureList = feature_extraction("cleaned_data.csv", featureDict, lang)
    # Split train/test samples
    depth_range = range(1, 5, 1)
    for i in depth_range:
        X_train, X_test, y_train, y_test = train_test_split(cleanX0, cleanY0, test_size=split, random_state=42)
        clf_tree = DecisionTreeClassifier(max_depth = i, random_state = 42)
        er_tree = generic_clf(X_train, y_train, X_test, y_test, clf_tree)
        er_train, er_test = [er_tree[0]], [er_tree[1]]
        print("splitPhase range: {}".format(i))
        errs = train(X_train, y_train, X_test, y_test, itr, clf_tree)
        er_train.append(errs[0])
        er_test.append(errs[1])
    pickle.dump(er_train, open("er_train_{}_{}_Clean.pkl".format(lang, itr), 'wb'))
    pickle.dump(er_test, open("er_test_{}_{}_Clean.pkl".format(lang, itr), 'wb'))
    plot_error_rate_depth(er_train, er_test, lang, 5)

def genTestResultFile(lang, idealSplitSize, idealNboost, idealTreeDept):
    cleanX0, cleanY0, trainFeatureList = feature_extraction("cleaned_data.csv", pub_feature_dict, lang)
    X_train, X_test, y_train, y_test = train_test_split(cleanX0, cleanY0, test_size=idealSplitSize, random_state=42)
    clf_tree = DecisionTreeClassifier(max_depth = idealTreeDept, random_state = 42)
    trainResults = training(X_train, y_train, idealNboost, clf_tree)
    clsTrainedList = trainResults[0]
    clsAlphaTrainedList = trainResults[1]
    labelledResult = test(testArr, clsTrainedList, clsAlphaTrainedList, idealNboost, lang)
    pickle.dump(clsTrainedList, open("clsTrainedList{}_{}_Clean.pkl".format(lang, idealNboost), 'wb'))
    pickle.dump(clsAlphaTrainedList, open("clsAlphaTrainedList{}_{}_Clean.pkl".format(lang, idealNboost), 'wb'))

def combinePhase(itr0, itr1, itr2, itr3, itr4):
# weight results
    test_set_y_0_file = "test_set_y_0_{}.csv".format(itr0)
    test_set_y_1_file = "test_set_y_1_{}.csv".format(itr1)
    test_set_y_2_file = "test_set_y_2_{}.csv".format(itr2)
    test_set_y_3_file = "test_set_y_3_{}.csv".format(itr3)
    test_set_y_4_file = "test_set_y_4_{}.csv".format(itr4)
    df0 = pd.read_csv(test_set_y_0_file, encoding="utf8")
    df1 = pd.read_csv(test_set_y_1_file, encoding="utf8")
    df2 = pd.read_csv(test_set_y_2_file, encoding="utf8")
    df3 = pd.read_csv(test_set_y_3_file, encoding="utf8")
    df4 = pd.read_csv(test_set_y_4_file, encoding="utf8")
    weight_0_list, weight_1_list, weight_2_list, weight_3_list, weight_4_list = df0['weight'], df1['weight'], df2['weight'], df3['weight'], df4['weight']

    label_list = []
    label_weight_list = []
    totalCount = len(weight_0_list)
    curCount = 0
    while curCount < totalCount:
        curList = [weight_0_list[curCount], weight_1_list[curCount], weight_2_list[curCount], weight_3_list[curCount], weight_4_list[curCount]]
        max_idx = np.argmax(curList)
        max_weight = max(curList)
        label_list.append(max_idx)
        label_weight_list.append(max_weight)
        curCount = curCount + 1
        percentComplete_test_final = curCount * 100 / totalCount
        print("Final Test Labelling Percent Complete: {}%".format(round(percentComplete_test_final, 2)))

    dataLabelWeight = {'label': label_list, 'weight': label_weight_list}
    label_list_df = pd.DataFrame(dataLabelWeight)

    label_list_df.to_csv("clean_test_set_y_{}_{}_{}_{}_{}.csv".format(itr0, itr1, itr2, itr3, itr4), encoding="utf8")

    empty_slot_lists_df = pd.read_csv("test_empty_slots.csv", encoding="utf8")
    empty_list = empty_slot_lists_df['empty_slot']

    # fill back empty slots to French cuz it contributes more in training dataset
    for i in empty_list:
        label_list[i] = 1

    new_labels_df = pd.DataFrame(label_list, columns=['new_label'])
    new_labels_df.to_csv("clean_test_set_y_with_empty_slot_refill_labels_{}_{}_{}_{}_{}.csv".format(itr0, itr1, itr2, itr3, itr4), encoding="utf8")


def skAdaboost():
    # Get features
    testArr, pub_feature_dict = feature_extraction_test("test_set_x.csv")
    cleanX0, cleanY0, trainFeatureList = feature_extraction_for_sklearn("cleaned_data.csv", pub_feature_dict)
    X_train, X_test, y_train, y_test = train_test_split(cleanX0, cleanY0, test_size=0.35, random_state=42)
    kfold = model_selection.KFold(n_splits=10, random_state=42)
    model = AdaBoostClassifier(n_estimators=70, random_state=42)
    clf = model.fit(X_train, y_train)

    test_y = clf.predict(testArr)
    pred_test_y_df = pd.DataFrame(test_y, columns=['label'])
    pred_test_y_df.to_csv("test_set_y_skboost.csv", encoding="utf8")

    empty_slot_lists_df = pd.read_csv("test_empty_slots.csv", encoding="utf8")
    empty_list = empty_slot_lists_df['empty_slot']
    # fill back empty slots to French cuz it contributes more in training dataset
    for i in empty_list:
        test_y[i] = 1

    new_labels_df = pd.DataFrame(test_y, columns=['Category'])
    new_labels_df.to_csv("sk_adaboost.csv", encoding="utf8")

def skAdaboostAccuracy():
    clf_tree = DecisionTreeClassifier(max_depth = 2, random_state = 42)
    testArr, pub_feature_dict = feature_extraction_test("test_set_x.csv")
    cleanX0, cleanY0, trainFeatureList = feature_extraction_for_sklearn("cleaned_data.csv", pub_feature_dict)
    X_train, X_test, y_train, y_test = train_test_split(cleanX0, cleanY0, test_size=0.35, random_state=42)
    # Fit a simple decision tree first    
    # Test with different number of iterations
    scoreList = []
    itr_range = range(1, 100, 10)
    for i in itr_range: 
        print("range: {}".format(i))
        model = AdaBoostClassifier(n_estimators=i, random_state=42)
        clf = model.fit(X_train, y_train)
        kfold = model_selection.KFold(n_splits=10, random_state=42)
        testing_score = cross_val_score(clf, X_test, y_test, cv=kfold)
        scoreList.append(testing_score.mean())
        print("Sk-learn AdaBoostClassifier testing accuracy: {}".format(testing_score.mean()))

    # Compare error rate vs number of iterations
    plot_score_sk(scoreList, 100)

#   ----main starts from here----
# part 1 adaboost sklearn
skAdaboostAccuracy()
skAdaboost()

# part 2 fully implemented adaboost
# Validation code
# idx = 1
# treeDept = 5
# iteration = 50
# splitPer = 0.35

# 1. Choose ideal parameter
# itrValidationPhase(0, 100, 20, pub_feature_dict, treeDept, splitPer)
# itrValidationPhase(1, 100, 20, pub_feature_dict, treeDept, splitPer)
# itrValidationPhase(2, 100, 20, pub_feature_dict, treeDept, splitPer)
# itrValidationPhase(3, 100, 20, pub_feature_dict, treeDept, splitPer)
# itrValidationPhase(4, 100, 20, pub_feature_dict, treeDept, splitPer)
# splitValidationPhase(idx, pub_feature_dict, treeDept, iteration)
# treeDeptValidationPhase(idx, pub_feature_dict, splitPer, iteration)

# 2. generate test_y.csv file for all 5 languages
# genTestResultFile(0, splitPer, 20, treeDept)
# genTestResultFile(1, splitPer, 40, treeDept)
# genTestResultFile(2, splitPer, 40, treeDept)
# genTestResultFile(3, splitPer, 20, treeDept)
# genTestResultFile(4, splitPer, 20, treeDept)

# 3. combine 5 test_y results into 1 multi-labelled file, specify the chosen iteration num for each lang
# combinePhase(20, 40, 40, 20, 20)

