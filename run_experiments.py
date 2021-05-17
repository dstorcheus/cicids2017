"""Runs experiments on CICIDS-2017 dataset."""
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
import sklearn
import tqdm
from tqdm import tqdm
from tqdm import tqdm_notebook
import xgboost as xgb
from incremental_trees.models.classification.streaming_rfc import StreamingRFC
import time
import tensorflow as tf
import sys
import matplotlib.pyplot as plt  # plotting
import numpy as np  # linear algebra
import os  # accessing directory structure
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from keras.models import Sequential
from keras.layers import Dense
import pickle as pkl
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
nRowsRead = None

# Some hardcoded parameters:
tf.compat.v1.flags.DEFINE_integer('sample', 10000, '')
tf.compat.v1.flags.DEFINE_boolean('notebook', False, '')
tf.compat.v1.flags.DEFINE_integer(
    'num_steps', 1, 'number of training step per new batch in online learning.')
tf.compat.v1.flags.DEFINE_integer(
    'n_batch_to_retrain', 1, 'number of old batch to retrain in online learning.')
tf.compat.v1.flags.DEFINE_integer('batch_size', 256, '')
tf.compat.v1.flags.DEFINE_string('run', '8,9,10,11', '')
FLAGS = tf.compat.v1.flags.FLAGS
progress_bar = tqdm
df_cache = None


def load_data(sampled_instances=100000):
    """Returns sampled cicids data as pd.df.
    Input: loads dataset from local memory.
    Output: processed dataframe.
    """

    global df_cache
    if df_cache is not None:
        return df_cache

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

    if sampled_instances > 0 and sampled_instances < nRow:
        df = df.sample(n=sampled_instances)

    df_cache = df
    return df


def preprocess_data_online(df):
    """Data preprocessing pipeline for online evaluation.
    Input: dataframe with cicids data
    Output: [training set, testing set]
    """
    train = df
    train.describe()
    # Packet Attack Distribution
    train[' Label'].value_counts()
    train = train.replace([np.inf, -np.inf], np.nan)
    train = train.dropna(how='all')
    # Scalling numerical attributes
    scaler = StandardScaler()
    # extract numerical attributes and scale it to have zero mean and unit variance
    cols = train.select_dtypes(include=['float64', 'int64']).columns
    sc_train = scaler.fit_transform(
        train.select_dtypes(include=['float64', 'int64']))
    # turn the result back to a dataframe
    sc_traindf = pd.DataFrame(sc_train, columns=cols)
    # creating one hot encoder object
    onehotencoder = OneHotEncoder()
    trainDep = train[' Label'].values.reshape(-1, 1)
    trainDep = onehotencoder.fit_transform(trainDep).toarray()
    # Scaled training data is prepared below.
    train_X = sc_traindf
    train_y = trainDep[:, 0]
    # Remove NaN from train and test
    train_X = train_X.replace([np.inf, -np.inf], np.nan)
    train_X = train_X.dropna(how='all')
    return train_X, train_y


def preprocess_data(df, test_size_=0.3, split_type='sequential'):
    """Returns train and test data.
    Input:
     - dataframe with cicids data
     - ratio of testing set size
    Output: train_X, train_y, test_X, test_y
    X stands for features and y for labels (intrusion/benign)
    """
    # Split dataset on train and test.
    if split_type == 'random':
        train, test = train_test_split(
            df, test_size=test_size_, random_state=10)
    if split_type == 'sequential':
        num_instances = len(df.index)
        train_length = int((1 - test_size_) * num_instances)
        train = df[:train_length]
        test = df[(train_length + 1):]
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
    # Remove NaN from train and test
    train_X = train_X.replace([np.inf, -np.inf], np.nan)
    test_X = test_X.replace([np.inf, -np.inf], np.nan)
    train_X = train_X.dropna(how='all')
    test_X = test_X.dropna(how='all')

    return train_X, train_y, test_X, test_y


def online_data_gen_with_retrain(
        predict_fn, pred_list,
        train_x, train_y,
        batch_size=256,
        n_batch_to_retrain=1,
        num_steps=1,
        yield_minibatch=False,
        pretrain_epochs=1,
        train_split=0.7):
    """ Generates online dtaa using the followin algorithm:
    Algorithm:
     - With each new batch:
     - Train on it for num_steps
     - Together with n_batch_to_retrain old batches randomly sampled.
    """
    train_x = train_x.to_numpy()
    # train_y = train_y.to_numpy()

    n = train_x.shape[0]
    m = int(n * train_split)
    m_range = np.arange(m)

    pretrain_x = train_x[:m, :]
    pretrain_y = train_y[:m]

    for _ in range(pretrain_epochs):

        if not yield_minibatch:
            yield pretrain_x, pretrain_y
            continue

        for _ in range(m // batch_size):
            # batchsize random numbers in [0 .. i-1]
            random_idx = np.random.choice(m_range, batch_size)
            yield pretrain_x[random_idx, :], pretrain_y[random_idx]

    i = m
    while i < n:

        if not yield_minibatch:
            yield train_x[:i + batch_size, :], train_y[:i + batch_size]

        new_batch_x = train_x[i:i + batch_size, :]
        new_batch_y = train_y[i:i + batch_size]

        # Online progressive cross-validation:
        new_pred = predict_fn(new_batch_x)
        pred_list += list(new_pred)

        i += batch_size
        if i <= batch_size:
            continue  # will not do any retraining.

        if i >= n:
            break  # end of data.

        idx = np.arange(i)  # [0..i-1]
        for _ in range(num_steps):  # Repeat this num_steps times

            to_train_x = new_batch_x
            to_train_y = new_batch_y

            # Concatenate n_batch_to_retrain random old batches
            # to the new_batch
            for _ in range(n_batch_to_retrain):
                # batchsize random numbers in [0 .. i-1]
                random_idx = np.random.choice(idx, batch_size)
                retrain_x = train_x[random_idx, :]
                retrain_y = train_y[random_idx]

                to_train_x = np.concatenate([to_train_x, retrain_x], axis=0)
                to_train_y = np.concatenate([to_train_y, retrain_y], axis=0)

            if not yield_minibatch:
                yield to_train_x, to_train_y
                continue

            # Now we shuffle & yield n_batch_to_retrain+1 batches:
            shuffle_idx = np.arange(to_train_x.shape[0])
            np.random.shuffle(shuffle_idx)
            for j in range(n_batch_to_retrain + 1):

                from_idx = j * batch_size
                to_idx = from_idx + batch_size
                idx_to_yield = shuffle_idx[from_idx: to_idx]

                yield (to_train_x[idx_to_yield, :],
                       to_train_y[idx_to_yield])

        # So in total, we have yielded
        # (n_batch_to_retrain+1)*num_steps batches
        # for each new batch, in which the new batch
        # of data is yielded num_steps times.


def make_online_tf_dataset(
        predict_fn, pred_list,
        # model.predict() will be used on each new data batch
        # and the prediction will be concat to list `pred`
        # for progressive evaluation
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.153.3925&rep=rep1&type=pdf

        train_X_values, train_y,
        batch_size=256,
        n_batch_to_retrain=1,
        num_steps=1,
        num_pretrain_epochs=10):
    """Returns a tf dataset that give batches in online learning manner."""

    def callable_generator():
        for datum in online_data_gen_with_retrain(
                # used for progressive cross-validation.
                predict_fn, pred_list,
                train_X_values, train_y,
                batch_size,
                n_batch_to_retrain,
                num_steps,
                yield_minibatch=True,
                pretrain_epochs=num_pretrain_epochs):
            yield datum

    return tf.data.Dataset.from_generator(
        callable_generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, None), dtype=tf.float64),
            tf.TensorSpec(shape=(batch_size), dtype=tf.int32)))


def eval_auc(true_y, pred):
    """Evaluates AUC."""
    fpr, tpr, thresholds = metrics.roc_curve(true_y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    # print("F1")
    # print(f1_score(test_y, pred, average='macro'))
    return auc


def eval_acc(true_y, pred):
    """Evaluate accuracy."""
    pred = (np.array(pred) >= 0.5).astype(np.float64)
    return np.sum(np.equal(true_y, pred)) * 1.0 / len(pred)


def train_dnn_online(train_X, train_y,
                     batch_size=256,
                     n_batch_to_retrain=1,
                     num_steps=1,
                     num_pretrain_epochs=10):
    """Trains DNN model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    input_dim_ = len(list(train_X.columns))
    model = Sequential()
    model.add(Dense(12, input_dim=input_dim_, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    # fit the keras model on the dataset

    total_steps = train_X.shape[0] // batch_size
    total_steps *= num_steps * (n_batch_to_retrain + 1)
    print('Total: {} steps'.format(total_steps))

    pred = []  # predictions on new batch will be concat here

    def predict_fn(x):
        return model.predict(x).flatten()

    model.fit(
        make_online_tf_dataset(
            predict_fn, pred, train_X, train_y,
            batch_size, n_batch_to_retrain, num_steps,
            num_pretrain_epochs=num_pretrain_epochs),
        epochs=1)

    train_y = train_y[-len(pred):]
    acc = eval_acc(train_y, pred)
    auc = eval_auc(train_y, pred)

    print("DNN ACC: {}".format(acc))
    print("DNN AUC: {}".format(auc))


def train_dnn(train_X, train_y, test_X, test_y):
    """Trains DNN model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
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


def train_rf_online(train_X, train_y,
                    batch_size=256,
                    n_batch_to_retrain=1,
                    num_steps=1):
    """Trains rf model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    srfc = StreamingRFC(
        n_estimators_per_chunk=20,
        max_n_estimators=np.inf,
        n_jobs=4)

    total_steps = train_X.shape[0] // batch_size
    total_steps *= num_steps
    print('Total {} steps'.format(total_steps))

    rfe = [None]

    def predict_fn(x):
        if rfe[0] is None:
            return [0.5] * x.shape[0]
        probs = rfe[0].predict_proba(x)
        return probs[:, 1]

    pred = []
    datagen = online_data_gen_with_retrain(
        predict_fn, pred, train_X, train_y,
        batch_size, n_batch_to_retrain, num_steps)

    for x, y in progress_bar(datagen, total=total_steps):
        rfe[0] = srfc.partial_fit(x, y, classes=[0, 1])

    train_y = train_y[-len(pred):]
    auc = eval_auc(train_y, pred)
    acc = eval_acc(train_y, pred)
    print("RF AUC: {}".format(auc))
    print("RF ACC: {}".format(acc))


def train_rf(train_X, train_y, test_X, test_y):
    """Trains rf model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    rfc = RandomForestClassifier()
    rfe = rfc.fit(train_X, train_y)
    pred = rfe.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("RF AUC: {}".format(auc))


def train_xgb(train_X, train_y, test_X, test_y):
    """Trains xgb model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
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
    """Trains xgb model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    KNN_Classifier = KNeighborsClassifier(n_jobs=-1)
    KNN_Classifier.fit(train_X, train_y)
    pred = KNN_Classifier.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("KNN AUC: {}".format(auc))


def train_lgr_online(train_X, train_y, batch_size=256,
                     n_batch_to_retrain=1, num_steps=1):
    """Trains logistic regression model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)

    total_steps = train_X.shape[0] // batch_size
    total_steps *= num_steps
    print('Total {} steps'.format(total_steps))

    def predict_fn(x):
        try:
            return LGR_Classifier.predict_proba(x)[:, 1]
        except sklearn.exceptions.NotFittedError:
            return [0.5] * x.shape[0]

    pred = []
    datagen = online_data_gen_with_retrain(
        predict_fn, pred, train_X, train_y,
        batch_size, n_batch_to_retrain, num_steps)

    for x, y in progress_bar(datagen, total=total_steps):
        LGR_Classifier.partial_fit(x, y, classes=[0, 1])

    auc = eval_auc(train_y, pred)
    acc = eval_acc(train_y, pred)

    print("LGR AUC: {}".format(auc))
    print("LGR ACC: {}".format(acc))


def train_lgr(train_X, train_y, test_X, test_y):
    """Trains logistic regression model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    LGR_Classifier = LogisticRegression(n_jobs=-1, random_state=0)
    LGR_Classifier.fit(train_X, train_y)
    pred = LGR_Classifier.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("LGR AUC: {}".format(auc))


def train_bnb_online(train_X, train_y, batch_size=256,
                     n_batch_to_retrain=1, num_steps=1):
    """Trains Bernoully  model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    BNB_Classifier = BernoulliNB()

    total_steps = train_X.shape[0] // batch_size
    total_steps *= num_steps
    print('Total {} steps'.format(total_steps))

    def predict_fn(x):
        try:
            return BNB_Classifier.predict_proba(x)[:, 1]
        except sklearn.exceptions.NotFittedError:
            return [0.5] * x.shape[0]

    pred = []
    datagen = online_data_gen_with_retrain(
        predict_fn, pred, train_X, train_y,
        batch_size, n_batch_to_retrain, num_steps)

    for x, y in progress_bar(datagen, total=total_steps):
        BNB_Classifier.partial_fit(x, y, classes=[0, 1])

    train_y = train_y[-len(pred):]
    auc = eval_auc(train_y, pred)
    acc = eval_acc(train_y, pred)
    print("BNB AUC: {}".format(auc))
    print("BNB ACC: {}".format(acc))


def train_bnb(train_X, train_y, test_X, test_y):
    """Trains Bernoully  model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    BNB_Classifier = BernoulliNB()
    BNB_Classifier.fit(train_X, train_y)
    pred = BNB_Classifier.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    acc = eval_acc(test_y, pred_)
    print("BNB AUC: {}".format(auc))
    print("BNB ACC: {}".format(acc))


def train_dtc_online(train_X, train_y,
                     batch_size=256,
                     n_batch_to_retrain=1,
                     num_steps=1):
    """Trains rf model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    srfc = StreamingRFC(
        n_estimators_per_chunk=1,
        max_features=train_X.shape[1],
        n_jobs=4)

    total_steps = train_X.shape[0] // batch_size
    total_steps *= num_steps
    print('Total {} steps'.format(total_steps))

    rfe = [None]

    def predict_fn(x):
        if rfe[0] is None:
            return [0.5] * x.shape[0]
        probs = rfe[0].predict_proba(x)
        return probs[:, 1]

    pred = []
    datagen = online_data_gen_with_retrain(
        predict_fn, pred, train_X, train_y,
        batch_size, n_batch_to_retrain, num_steps)

    for x, y in progress_bar(datagen, total=total_steps):
        rfe[0] = srfc.partial_fit(x, y, classes=[0, 1])

    train_y = train_y[-len(pred):]
    auc = eval_auc(train_y, pred)
    acc = eval_acc(train_y, pred)
    print("DTC AUC: {}".format(auc))
    print("DTC ACC: {}".format(acc))


def train_dtc(train_X, train_y, test_X, test_y):
    """Trains Decision tree model.
    Input: model parameters
    Output: saves model to memory and prints AUC.
    """
    DTC_Classifier = tree.DecisionTreeClassifier(
        criterion='entropy', random_state=0)
    DTC_Classifier.fit(train_X, train_y)
    pred = DTC_Classifier.predict_proba(test_X.values)
    pred_ = []
    for x in pred:
        pred_.append(x[1])
    auc = eval_auc(test_y, pred_)
    print("DTC AUC: {}".format(auc))


def select_features(train_X, train_y, test_X=None, test_y=None, k=20):
    """Selects k features using Random forest.
    Input: training and testing dataset
    Output: sequence of selected features
    """

    selected_feat_fname = 'selected.{}.pkl'.format(k)
    if not os.path.exists(selected_feat_fname):
        # Recursive feature elimination
        rfc = RandomForestClassifier()
        # create the RFE model and select 20 attributes
        rfe = RFE(rfc, n_features_to_select=20)
        rfe = rfe.fit(train_X, train_y)
        # summarize the selection of the attributes
        feature_map = [(i, v) for i, v in itertools.zip_longest(
            rfe.get_support(), train_X.columns)]

        with open(selected_feat_fname, 'wb') as f:
            pkl.dump(feature_map, f)
    else:
        print('Loading {}'.format(selected_feat_fname))
        with open(selected_feat_fname, 'rb') as f:
            feature_map = pkl.load(f)

    # selected_features = [v for i, v in feature_map if i == True]
    # print('Selected Features')
    # print(selected_features)

    a = [i[0] for i in feature_map]
    train_X = train_X.iloc[:, a]

    if test_X is not None:
        test_X = test_X.iloc[:, a]
        return train_X, train_y, test_X, test_y

    return train_X, train_y

# The helper functions below are used to run various experiments.
# Call them in the main file.

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


def run_experiment_8():
    """DNN online, with and without feature selection."""
    df = load_data(FLAGS.sample)
    train_X, train_y = preprocess_data_online(df)

    print("DNN without feature selection.")
    for pretrain_epochs in [1, 2, 5, 10, 15, 20]:
        print("Train DNN online with pretrain_epochs = {}.".format(pretrain_epochs))
        train_dnn_online(
            train_X, train_y,
            FLAGS.batch_size, FLAGS.n_batch_to_retrain, FLAGS.num_steps,
            pretrain_epochs)
        print("DNN with feature selection.")
        train_X_selected, train_y_selected = select_features(train_X, train_y)
        train_dnn_online(
            train_X_selected, train_y_selected,
            FLAGS.batch_size, FLAGS.n_batch_to_retrain, FLAGS.num_steps,
            pretrain_epochs)


def run_experiment_9():
    """Runs rf with and without feature selection."""
    print("Train RF online.")
    df = load_data(FLAGS.sample)
    train_X, train_y = preprocess_data_online(df)
    print("RF without feature selection.")
    train_rf_online(
        train_X, train_y,
        FLAGS.batch_size, FLAGS.n_batch_to_retrain, FLAGS.num_steps)
    train_X, train_y = select_features(
        train_X, train_y)
    print("RF with feature selection.")
    train_rf_online(
        train_X, train_y,
        FLAGS.batch_size, FLAGS.n_batch_to_retrain, FLAGS.num_steps)


def run_experiment_10():
    """Runs BNB with and without feature selection."""
    print("Train BNB online.")
    df = load_data(FLAGS.sample)
    train_X, train_y = preprocess_data_online(df)
    print("BNB without feature selection.")
    train_bnb_online(
        train_X, train_y,
        FLAGS.batch_size, FLAGS.n_batch_to_retrain, FLAGS.num_steps)
    train_X, train_y = select_features(train_X, train_y)
    print("BNB with feature selection.")
    train_bnb_online(
        train_X, train_y,
        FLAGS.batch_size, FLAGS.n_batch_to_retrain, FLAGS.num_steps)


def run_experiment_11():
    """Runs DTC with and without feature selection."""
    print("Train DTC online.")
    df = load_data(FLAGS.sample)
    train_X, train_y = preprocess_data_online(df)
    print("DTC without feature selection.")
    train_dtc_online(
        train_X, train_y,
        FLAGS.batch_size, FLAGS.n_batch_to_retrain, FLAGS.num_steps)
    train_X, train_y = select_features(
        train_X, train_y)
    print("DTC with feature selection.")
    train_dtc_online(
        train_X, train_y,
        FLAGS.batch_size, FLAGS.n_batch_to_retrain, FLAGS.num_steps)


# def run_experiment_12():
#     """Runs LGR with and without feature selection."""
#     print("Train LGR online.")
#     df = load_data()
#     train_X, train_y = preprocess_data_online(df)
#     print("LGR without feature selection.")
#     train_lgr_online(train_X, train_y)
#     train_X, train_y = select_features(train_X, train_y)
#     print("LGR with feature selection.")
#     train_lgr_online(train_X, train_y)


if __name__ == '__main__':
    print('Run at time stamp ' + str(time.time()))
    """The main file executes experiments.
    To run experiments, call the necessary function
    run_experiment_k() for k from 1 to 12.

    Each experiment trains a specific type of AI model,
    including DNN, Decision Trees, Naive Bayes, etc.

    Every experiment has bpth online and offline version of
    the model evaluation and outouts AUC metric.

    """
    if FLAGS.notebook:
        progress_bar = tqdm_notebook

    # run_experiment_5()
    # run_experiment_6()
    # run_experiment_7()

    if FLAGS.run:
        for exp_id in FLAGS.run.split(','):
            eval('run_experiment_' + exp_id)()
    else:  # manual call
        run_experiment_8()
        # run_experiment_9()
        run_experiment_10()
        run_experiment_11()
        # run_experiment_12()
