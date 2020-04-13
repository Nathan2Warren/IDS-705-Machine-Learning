# -*- coding: utf-8 -*-
'''Sample script for solar array image classification

Author:       Kyle Bradbury
Date:         January 30, 2018
Organization: Duke University Energy Initiative
'''

'''
Import the packages needed for classification
'''

from skimage.color import rgb2grey
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as metrics
from skimage.filters import prewitt_h, prewitt_v, prewitt
from sklearn import svm
plt.close()
os.chdir('/Users/N1/Desktop/705 - ML/Kaggle/')
'''
Set directory parameters
'''
# Set the directories for the data and the CSV files that contain ids/labels
dir_train_images = './data/training/'
dir_test_images = './data/testing/'
dir_train_labels = './data/labels_training.csv'
dir_test_ids = './data/sample_submission.csv'

'''
Include the functions used for loading, preprocessing, features extraction,
classification, and performance evaluation
'''


def load_data(dir_data, dir_labels, training=True):
    ''' Load each of the image files into memory

    While this is feasible with a smaller dataset, for larger datasets,
    not all the images would be able to be loaded into memory

    When training=True, the labels are also loaded
    '''
    labels_pd = pd.read_csv(dir_labels)
    ids = labels_pd.id.values
    data = []
    for identifier in ids:
        fname = dir_data + identifier.astype(str) + '.tif'
        image = mpl.image.imread(fname)
        data.append(image)
    data = np.array(data)  # Convert to Numpy array
    if training:
        labels = labels_pd.label.values
        return data, labels
    else:
        return data, ids


def preprocess_and_extract_features(data):
    '''Preprocess data and extract features

    Preprocess: normalize, scale, repair
    Extract features: transformations and dimensionality reduction
    '''
    # Here, we do something trivially simple: we take the average of the RGB
    # values to produce a grey image, transform that into a vector, then
    # extract the mean and standard deviation as features.

    # Make the image grayscale
    #data = np.mean(data, axis=3)
    dataR = data[:, :, :, 0]
    data1 = []
    for row in range(dataR.shape[0]):
        data_4loop = prewitt(dataR[row])
        data1.append(data_4loop)
    dataR = np.array(data1)
    #np.clip(data, 0, 170, out=data)
    # Vectorize the grayscale matrices
    vectorized_data = dataR.reshape(dataR.shape[0], -1)

    # extract the mean and standard deviation of each sample as features
    feature_mean = np.mean(vectorized_data, axis=1)
    feature_std = np.std(vectorized_data, axis=1)

    # Combine the extracted features into a single feature vector
    featuresR = np.stack((feature_mean, feature_std), axis=-1)

    dataG = data[:, :, :, 1]
    data1 = []
    for row in range(dataG.shape[0]):
        data_4loop = prewitt(dataG[row])
        data1.append(data_4loop)
    dataG = np.array(data1)
    #np.clip(data, 0, 170, out=data)
    # Vectorize the grayscale matrices
    vectorized_data = dataG.reshape(dataG.shape[0], -1)

    # extract the mean and standard deviation of each sample as features
    feature_mean = np.mean(vectorized_data, axis=1)
    feature_std = np.std(vectorized_data, axis=1)
    feature_mean.shape

    # Combine the extracted features into a single feature vector
    featuresG = np.stack((feature_mean, feature_std), axis=-1)

    dataB = data[:, :, :, 2]
    data1 = []
    for row in range(dataB.shape[0]):
        data_4loop = prewitt(dataB[row])
        data1.append(data_4loop)
    dataB = np.array(data1)
    #np.clip(data, 0, 170, out=data)
    # Vectorize the grayscale matrices
    vectorized_data = dataB.reshape(dataB.shape[0], -1)

    # extract the mean and standard deviation of each sample as features
    feature_mean = np.mean(vectorized_data, axis=1)
    feature_std = np.std(vectorized_data, axis=1)

    # Combine the extracted features into a single feature vector
    featuresB = np.stack((feature_mean, feature_std), axis=-1)
    features = np.vstack((featuresR, featuresG))

    return features


def set_classifier():
    '''Shared function to select the classifier for both performance evaluation
    and testing
    '''
    return svm.SVC(probability=True)


def cv_performance_assessment(X, y, k, clf):
    '''Cross validated performance assessment

    X   = training data
    y   = training labels
    k   = number of folds for cross validation
    clf = classifier to use

    Divide the training data into k folds of training and validation data.
    For each fold the classifier will be trained on the training data and
    tested on the validation data. The classifier prediction scores are
    aggregated and output
    '''
    # Establish the k folds
    prediction_scores = np.empty(y.shape[0], dtype='object')
    kf = StratifiedKFold(n_splits=k, shuffle=True)
    for train_index, val_index in kf.split(X, y):
        # Extract the training and validation data for this fold
        X_train, X_val = X[train_index], X[val_index]
        y_train = y[train_index]

        # Train the classifier
        X_train_features = preprocess_and_extract_features(X_train)
        clf = clf.fit(X_train_features, y_train)

        # Test the classifier on the validation data for this fold
        X_val_features = preprocess_and_extract_features(X_val)
        cpred = clf.predict_proba(X_val_features)

        # Save the predictions for this fold
        prediction_scores[val_index] = cpred[:, 1]
    return prediction_scores


def plot_roc(labels, prediction_scores):
    fpr, tpr, _ = metrics.roc_curve(labels, prediction_scores, pos_label=1)
    auc = metrics.roc_auc_score(labels, prediction_scores)
    legend_string = 'AUC = {:0.3f}'.format(auc)

    plt.plot([0, 1], [0, 1], '--', color='gray', label='Chance')
    plt.plot(fpr, tpr, label=legend_string)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid('on')
    plt.axis('square')
    plt.legend()
    plt.tight_layout()


'''
Sample script for cross validated performance
'''
# Set parameters for the analysis
num_training_folds = 20

# Load the data
data, labels = load_data(dir_train_images, dir_train_labels, training=True)

# Choose which classifier to use
clf = set_classifier()

# Perform cross validated performance assessment
prediction_scores = cv_performance_assessment(data, labels, num_training_folds, clf)

# Compute and plot the ROC curves
plot_roc(labels, prediction_scores)


'''
Sample script for producing a Kaggle submission
'''

produce_submission = False  # Switch this to True when you're ready to create a submission for Kaggle

if produce_submission:
    # Load data, extract features, and train the classifier on the training data
    training_data, training_labels = load_data(dir_train_images, dir_train_labels, training=True)
    training_features = preprocess_and_extract_features(training_data)
    clf = set_classifier()
    clf.fit(training_features, training_labels)

    # Load the test data and test the classifier
    test_data, ids = load_data(dir_test_images, dir_test_ids, training=False)
    test_features = preprocess_and_extract_features(test_data)
    test_scores = clf.predict_proba(test_features)[:, 1]

    # Save the predictions to a CSV file for upload to Kaggle
    submission_file = pd.DataFrame({'id':    ids,
                                    'score':  test_scores})
    submission_file.to_csv('submission.csv',
                           columns=['id', 'score'],
                           index=False)