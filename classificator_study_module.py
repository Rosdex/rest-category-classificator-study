import numpy as np
import pandas as pd
import os
import csv

from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.utils import shuffle

#wrapper for Tomita parser
from tomita_parser import TomitaParser

UPLOAD_DIR = 'uploads'
RESULT_DIR = 'results'
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TOMITA_BIN_PATH = '\\'.join([BASE_PATH, 'tomita', 'tomitaparser.exe'])
TOMITA_CONFIG_PATH = '\\'.join([BASE_PATH, 'tomita', 'config', 'config.proto'])

class CategoryClassificatorStudy:

    def __init__(self):
        # prepare Tomita Parser
        self.tomita = TomitaParser(TOMITA_BIN_PATH, TOMITA_CONFIG_PATH, debug=False)
        self.vectorizer = None
        self.ml_model = None
        self.model_acc = 0

    def train_classificator(self, dataset_filename):
        # Step 0 - read dataset
        product_names, product_labels = self.read_dataset(dataset_filename)

        # Step 1 - clean name using Tomita parser
        # WARNING Here will be problem with execptions, refactor this part
        product_names, product_labels = self.clean_names(product_names, product_labels)

        # Step 2 - create vectorizer
        self.vectorizer = self.create_vectorizer(product_names)

        # Step 3 - perform names vectorization
        product_names = self.names_vectorization(product_names)

        # Step 4 - Shuffle Dataset
        # WARNING Shape should be same
        product_names, product_labels = shuffle(product_names, product_labels)

        # Step 5 - Split dataset to Train and Test
        # Step 5.1 - Calculate sizes
        dataset_len = len(product_names)
        train_size, test_size = self.calculate_train_test_sizes(dataset_len)
        
        # Step 5.2 - Create train dataset 
        train_X, train_Y = self.create_train_dataset(product_names, product_labels, train_size)

        # Step 5.3 - Create test dataset
        test_X, test_Y = self.create_test_dataset(product_names, product_labels, train_size, dataset_len)

        # Step 6 - Create SVM model
        self.ml_model = self.create_svm_model(train_X, train_Y)

        # Step 7 - Validate model
        self.model_acc = self.validate_model(test_X, test_Y)

        print('Classificator trained')
        return 0

    def read_dataset(self, input_file):
        filename = '\\'.join([UPLOAD_DIR, input_file])
        names = []
        labels = []
    
        with open(filename, newline='', encoding="utf8") as csvfile:
            csvreader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in csvreader:
                names.append(row[0])
                labels.append(row[1])

        return names, labels

    def clean_names(self, raw_names, labels):
        cleaned_names = []
        new_labels = []

        for i in range(0, len(raw_names)):
            name = raw_names[i]
            label = labels[i]

            facts, leads = self.tomita.run(name)

            if(len(facts) > 0):
                cleaned_names.append(self.fact_to_string(facts[0]))
                new_labels.append(label)

        return cleaned_names, new_labels

    def create_vectorizer(self, names):
        # create the transform
        vectorizer = TfidfVectorizer(stop_words='english')

        # tokenize and build vocab
        vectorizer.fit(names)

        return vectorizer

    def names_vectorization(self, names):
        name_vectors = []

        for product_name in names:
            vector = self.vectorizer.transform([product_name])
            np_vector = self.create_np_array_from_vector(vector)
            name_vectors.append(np_vector)

        return name_vectors

    def calculate_train_test_sizes(self, dataset_len):
        train_size = int(dataset_len * 0.80)
        test_size = dataset_len - train_size
        print('train size = {0}, test size = {1}'.format(train_size, test_size))
        return train_size, test_size

    def create_train_dataset(self, names, labels, size):
        train_X = names[0:size]
        train_Y = labels[0:size]
        return train_X, train_Y

    def create_test_dataset(self, names, labels, train_size, dataset_len):
        test_X = names[train_size:dataset_len]
        test_Y = labels[train_size:dataset_len]
        return test_X, test_Y

    def create_svm_model(self, train_X, train_Y):
        #create SVM nodel
        clf = svm.SVC(gamma=0.001, C=100.)
        #train model
        clf.fit(train_X, train_Y)
        return clf

    def validate_model(self, test_X, test_Y):
        #check model on test data
        predicted_Y = self.ml_model.predict(test_X)
        #compare predicted and expected output
        result = np.mean(predicted_Y == test_Y)
        print('Model accuracy = {0}'.format(result))

        return result

    def save_model_files(self, uuid):
        #Save vectorizer
        vectorizator_filename = '_'.join([uuid, 'vectorizator.sav'])
        joblib.dump(self.vectorizer, '\\'.join([RESULT_DIR, vectorizator_filename]))

        #Save classification model
        model_filename = '_'.join([uuid, 'svm.sav'])
        joblib.dump(self.ml_model, '\\'.join([RESULT_DIR, model_filename]))

        return vectorizator_filename, model_filename

    def fact_to_string(self, fact):
        result_str = "{0} {1} ".format(fact['fact'], fact['adjForName'])
        result_str = result_str.lower().strip()
        return result_str

    def create_np_array_from_vector(self, vector):
        vec_arr = vector.toarray()
        vec_list = []
    
        for i in range(0, vector.shape[1]):
            vec_list.append(vec_arr[0,i])
        
        return np.array(vec_list)

