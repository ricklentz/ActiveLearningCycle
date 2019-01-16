

# open the data
import pickle
X_train = pickle.load(  open('X_train.pkl', "rb" ) )
X_test = pickle.load(  open('X_test.pkl', "rb" ) )
Y_train = pickle.load(  open('Y_train.pkl', "rb" ) )
Y_test = pickle.load(  open('Y_test.pkl', "rb" ) )
X_test_reg_features = pickle.load(  open('X_test_reg_features.pkl', "rb" ) )
X_test_high_features = pickle.load(  open('X_test_high_features.pkl', "rb" ) )
X_test_reg_features = pickle.load(  open('X_test_reg_features.pkl', "rb" ) )
X_test_high_features = pickle.load(  open('X_test_high_features.pkl', "rb" ) )


import pymagnitude
#load pretrained model
pretrained_magnitude = r'/Users/rwl012/Downloads/GoogleNews-vectors-negative300.magnitude'
vectors = pymagnitude.Magnitude(pretrained_magnitude)

# setup speciality cleaning
def get_document_features(data_in,highlights=False,c=0.5):
    data_in = data_in.replace('<span class=\"active_text\">', '')
    data_in = data_in.replace('</span>', '')
    body = data_in.split(r'\n                                    ')[1]
    
    body = body.replace('\n', '')
    return np.mean(vectors.query(body.split(' ')), axis=(0))
    
def get_document_features_highlights(data_in,c=0.5):
    data_in = data_in.replace('<span class=\"active_text\">', '')
    data_in = data_in.replace('</span>', '')
    body = data_in.split(r'\n                                    ')[1]
    
    body = body.replace('\n', '')
    avg_vec = np.mean(vectors.query(body.split(' ')), axis=(0))
    high_text = data_in.split(r'\n                                    ')[0]
    high_text = high_text.replace('\n', '')
    high_avg_vec = np.mean(vectors.query(high_text.split(' ')), axis=(0))
    return (avg_vec, high_avg_vec *(c) + (1-c)*avg_vec)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np

# used to copy the list of training data for distict selection
from copy import deepcopy

# used to accumulate training text samples for simulating active learning
y_train_list_baseline = list()
y_train_list_highlights = list()

# used to accumulate training sample features
norm_feat_train_list = list()
high_feat_train_list = list()

# used to capture auc for baseline and comparison
norm_auc = list()
high_auc = list()

# setup a second pool of training text
number_of_training_samples = len(X_train)
X_train_highlights = deepcopy(X_train)

# seed the initial model
text_baseline_index = 0
text_highlight_index = 0

# simulated active learning
for idx in range(0,number_of_training_samples):
    X_train[text_baseline_index]
    X_train_highlights[text_highlight_index]
    
    avg_vec = get_document_features(text_baseline)
    high_avg_vec = get_document_features_highlights(text_highlight,True,0.5)
    
    norm_feat_train_list.append(avg_vec)
    high_feat_train_list.append(high_avg_vec)
    y_train_list.append(Y_train[idx])
    
    # if enough classes to fit
    #if len(set(y_train_list_baseline)) > 1:
    print(idx)
    reg_model = LogisticRegression().fit(np.vstack(norm_feat_train_list), y_train_list_baseline)
    high_model = LogisticRegression().fit(np.vstack(norm_feat_train_list), y_train_list_highlights)

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, reg_model.predict(X_test_reg_features))
    n_auc = auc(false_positive_rate, true_positive_rate)
    norm_auc.append( n_auc )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, high_model.predict(X_test_high_features))
    h_auc = auc(false_positive_rate, true_positive_rate)
    high_auc.append( h_auc )
            
with open('norm_auc.pkl', 'wb') as f:
    pickle.dump(norm_auc, f)
with open('high_auc.pkl', 'wb') as f:
    pickle.dump(high_auc, f)
