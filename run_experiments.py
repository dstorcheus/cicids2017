"""Runs experiments on CICIDS-2017 dataset."""
#from mpl_toolkits.mplot3d import Axes3D
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
#import xgboost as xgb
import sys
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None

# Some hardcodede parameters:
sampled_instances_ = 10000


def load_data(sampled_instances=10000):
    """Returns sampled cicids data as pd.df."""
    df1 = pd.read_csv("Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")
    df2 = pd.read_csv("Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv")
    df3 = pd.read_csv("Friday-WorkingHours-Morning.pcap_ISCX.csv")
    df4 = pd.read_csv("Monday-WorkingHours.pcap_ISCX.csv")
    df5 = pd.read_csv(
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")
    df6 = pd.read_csv("Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
    df7 = pd.read_csv("Tuesday-WorkingHours.pcap_ISCX.csv")
    df8 = pd.read_csv("Wednesday-workingHours.pcap_ISCX.csv")

    df = pd.concat([df1, df2])
    del df1, df2
    df = pd.concat([df, df3])
    del df3
    df = pd.concat([df, df4])
    del df4
    df = pd.concat([df, df5])
    del df5
    df = pd.concat([df, df6])
    del df6
    df = pd.concat([df, df7])
    del df7
    df = pd.concat([df, df8])
    del df8

    nRow, nCol = df.shape
    print(f'{nRow} rows & {nCol} cols')

    # Some columns have inf values.
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df.head()
    df = df.sample(n=sampled_instances)

    return df


def preprocess_data(df, test_size_=0.3):
    """Returns train and test data."""
    # Split dataset on train and test
    train, test = train_test_split(df, test_size=test_size_, random_state=10)
    train.describe()
    test.describe()
    # Packet Attack Distribution
    train[' Label'].value_counts()
    test[' Label'].value_counts()
    train = train.replace([np.inf, -np.inf], np.nan)
    train = train.dropna(how='all')
    test = test.replace([np.inf, -np.inf], np.nan)
    test = test.dropna(how='all')
    # Scalling numerical attributes
    scaler = StandardScaler()
    # extract numerical attributes and scale it to have zero mean and unit variance
    cols = train.select_dtypes(include=['float64', 'int64']).columns
    sc_train = scaler.fit_transform(
        train.select_dtypes(include=['float64', 'int64']))
    sc_test = scaler.fit_transform(
        test.select_dtypes(include=['float64', 'int64']))
    # turn the result back to a dataframe
    sc_traindf = pd.DataFrame(sc_train, columns=cols)
    sc_testdf = pd.DataFrame(sc_test, columns=cols)
    # creating one hot encoder object
    onehotencoder = OneHotEncoder()
    trainDep = train[' Label'].values.reshape(-1, 1)
    trainDep = onehotencoder.fit_transform(trainDep).toarray()
    testDep = test[' Label'].values.reshape(-1, 1)
    testDep = onehotencoder.fit_transform(testDep).toarray()
    # Scaled training data is prepared below.
    train_X = sc_traindf
    train_y = trainDep[:, 0]
    test_X = sc_testdf
    test_y = testDep[:, 0]
    """
    print('Train and test histogram')
    import matplotlib.pyplot as plt
    plt.hist(train_y)
    plt.show()

    plt.hist(test_y)
    plt.show()
    """
    # Remove NaN from train and test
    train_X = train_X.replace([np.inf, -np.inf], np.nan)
    test_X = test_X.replace([np.inf, -np.inf], np.nan)
    train_X = train_X.dropna(how='all')
    test_X = test_X.dropna(how='all')

    return train_X, train_y, test_X, test_y


def eval_auc(true_y, pred):
    """Evaluates AUC."""
    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # print("F1")
    # print(f1_score(test_y, pred, average='macro'))
    return auc


def train_dnn(train_X, train_y, test_X, test_y):
    """Trains DNN model."""
    # DNN model without feature selection.
    input_dim_ = len(list(train_X.columns))
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim_, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset
    model.fit(train_X.values, train_y, epochs=20, batch_size=256)
    # evaluate the keras model
    _, accuracy = model.evaluate(test_X.values, test_y)
    print('Accuracy: %.2f' % (accuracy * 100))
    pred = model.predict(test_X.values).flatten()
    auc = eval_auc(test_y, pred)
    print("DNN AUC: {}".format(auc))


def train_rf(train_X, train_y, test_X, test_y):
    """Trains rf model."""
    rfc = RandomForestClassifier()
    rfe = rfc.fit(train_X, train_y)
    pred = rfe.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("RF AUC: {}".format(auc))


def train_xgb(train_X, train_y, test_X, test_y):
    """Trains xgb model."""
    xg = xgb.XGBClassifier(colsample_bytree=0.3, learning_rate=0.1,
                           max_depth=5, alpha=10, n_estimators=10)
    xg = xg.fit(train_X, train_y)
    pred = xg.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("XGB AUC: {}".format(auc))


def train_knn(train_X, train_y, test_X, test_y):
    """Trains xgb model."""
    KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
    KNN_Classifier.fit(train_X, train_y)
    pred = KNN_Classifier.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("KNN AUC: {}".format(auc))


def train_lgr(train_X, train_y, test_X, test_y):
    """Trains logistic regression model."""
    LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
    LGR_Classifier.fit(train_X, train_y)
    pred = LGR_Classifier.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("LGR AUC: {}".format(auc))


def train_bnb(train_X, train_y, test_X, test_y):
    """Trains Bernoully  model."""
    BNB_Classifier = BernoulliNB()
    BNB_Classifier.fit(train_X, train_y)
    pred = BNB_Classifier.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("BNB AUC: {}".format(auc))


def train_dtc(train_X, train_y, test_X, test_y):
    """Trains Decision tree model."""
    DTC_Classifier = tree.DecisionTreeClassifier(
        criterion='entropy', random_state=0)
    DTC_Classifier.fit(train_X, train_y)
    pred = DTC_Classifier.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("DTC AUC: {}".format(auc))


def select_features(train_X, train_y, test_X, test_y, k=20):
    """Selects k features using Random forest."""
    # Recursive feature elimination
    rfc = RandomForestClassifier()
    # create the RFE model and select 20 attributes
    rfe = RFE(rfc, n_features_to_select=20)
    rfe = rfe.fit(train_X, train_y)
    # summarize the selection of the attributes
    feature_map = [(i, v) for i, v in itertools.zip_longest(
        rfe.get_support(), train_X.columns)]
    selected_features = [v for i, v in feature_map if i == True]

    print('Selected Features')
    print(selected_features)

    a = [i[0] for i in feature_map]
    train_X = train_X.iloc[:, a]
    test_X = test_X.iloc[:, a]

    return train_X, train_y, test_X, test_y


def run_experiment_1():
    """Runs dnn with and without feature selection."""
    print("Train DNN.")
    df = load_data()
    train_X, train_y, test_X, test_y = preprocess_data(df)
    print("DNN without feature selection.")
    train_dnn(train_X, train_y, test_X, test_y)
    train_X, train_y, test_X, test_y = select_features(
        train_X, train_y, test_X, test_y)
    print("DNN with feature selection.")
    train_dnn(train_X, train_y, test_X, test_y)


def run_experiment_2():
    """Runs rf with and without feature selection."""
    print("Train RF.")
    df = load_data()
    train_X, train_y, test_X, test_y = preprocess_data(df)
    print("RF without feature selection.")
    train_rf(train_X, train_y, test_X, test_y)
    train_X, train_y, test_X, test_y = select_features(
        train_X, train_y, test_X, test_y)
    print("RF with feature selection.")
    train_rf(train_X, train_y, test_X, test_y)


# def run_experiment_3():
#    """Runs xgb with and without feature selection."""
#    print("Train XGB.")
#    df = load_data()
#    train_X, train_y, test_X, test_y = preprocess_data(df)
#    print("XGB without feature selection.")
#    train_xgb(train_X, train_y, test_X, test_y)
#    train_X, train_y, test_X, test_y = select_features(
#        train_X, train_y, test_X, test_y)
#    print("XGB with feature selection.")
#    train_xgb(train_X, train_y, test_X, test_y)


def run_experiment_4():
    """Runs KNN with and without feature selection."""
    print("Train KNN.")
    df = load_data()
    train_X, train_y, test_X, test_y = preprocess_data(df)
    print("KNN without feature selection.")
    train_knn(train_X, train_y, test_X, test_y)
    train_X, train_y, test_X, test_y = select_features(
        train_X, train_y, test_X, test_y)
    print("KNN with feature selection.")
    train_knn(train_X, train_y, test_X, test_y)


def run_experiment_5():
    """Runs LGR with and without feature selection."""
    print("Train LGR.")
    df = load_data()
    train_X, train_y, test_X, test_y = preprocess_data(df)
    print("LGR without feature selection.")
    train_lgr(train_X, train_y, test_X, test_y)
    train_X, train_y, test_X, test_y = select_features(
        train_X, train_y, test_X, test_y)
    print("LGR with feature selection.")
    train_lgr(train_X, train_y, test_X, test_y)


def run_experiment_6():
    """Runs BNB with and without feature selection."""
    print("Train BNB.")
    df = load_data()
    train_X, train_y, test_X, test_y = preprocess_data(df)
    print("BNB without feature selection.")
    train_bnb(train_X, train_y, test_X, test_y)
    train_X, train_y, test_X, test_y = select_features(
        train_X, train_y, test_X, test_y)
    print("BNB with feature selection.")
    train_bnb(train_X, train_y, test_X, test_y)


def run_experiment_7():
    """Runs DTC with and without feature selection."""
    print("Train DTC.")
    df = load_data()
    train_X, train_y, test_X, test_y = preprocess_data(df)
    print("DTC without feature selection.")
    train_dtc(train_X, train_y, test_X, test_y)
    train_X, train_y, test_X, test_y = select_features(
        train_X, train_y, test_X, test_y)
    print("DTC with feature selection.")
    train_dtc(train_X, train_y, test_X, test_y)


run_experiment_5()
run_experiment_6()
run_experiment_7()
