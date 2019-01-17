# open the upstream data
import pickle
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np

Y_train = pickle.load(  open('Y_train.pkl', "rb" ) )
Y_test = pickle.load(  open('Y_test.pkl', "rb" ) )
X_test_reg_features = pickle.load(  open('X_test_reg_features.pkl', "rb" ) )
X_test_high_features = pickle.load(  open('X_test_high_features.pkl', "rb" ) )
X_train_reg_features = pickle.load(  open('X_train_reg_features.pkl', 'rb') )
X_train_high_features = pickle.load(  open('X_train_high_features.pkl', 'rb') )

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

import copy
Y_train_high = copy.deepcopy(Y_train)

# used to accumulate training text samples for simulating active learning
y_train_list_baseline = list()
y_train_list_highlights = list()

# used to accumulate training sample features
norm_feat_train_list = list()
high_feat_train_list = list()

# used to capture auc for baseline and comparison
norm_auc = list()
high_auc = list()

# seed the initial model
itemindex_0 = np.nonzero(Y_train == 0)[0][0]
norm_feat_train_list.append(X_train_reg_features[itemindex_0])
X_train_reg_features.pop(itemindex_0)
high_feat_train_list.append(X_train_high_features[itemindex_0])
X_train_high_features.pop(itemindex_0)

y_train_list_baseline.append(Y_train[itemindex_0])
Y_train = np.delete(Y_train, Y_train[itemindex_0])

y_train_list_highlights.append(Y_train_high[itemindex_0])
Y_train_high = np.delete(Y_train_high,Y_train_high[itemindex_0])

# let the first iteration 
baseline_queue_index = np.nonzero(Y_train == 1)[0][0]
highlight_queue_index = np.nonzero(Y_train_high == 1)[0][0]


print(type(Y_train))

# simulated active learning
for idx in range(0,len(Y_train)):
    # We add it to the simulated processed data, then remove it from the available training sample queue
    norm_feat_train_list.append(X_train_reg_features[baseline_queue_index])
    X_train_reg_features.pop(baseline_queue_index)
    high_feat_train_list.append(X_train_high_features[highlight_queue_index])
    X_train_high_features.pop(highlight_queue_index)
    
    y_train_list_baseline.append(Y_train[baseline_queue_index])
    Y_train = np.delete(Y_train, Y_train[baseline_queue_index])

    y_train_list_highlights.append(Y_train_high[highlight_queue_index])
    Y_train_high = np.delete(Y_train_high,Y_train_high[highlight_queue_index])
    
    # We now have one of each 
    reg_model = LogisticRegression().fit(np.vstack(norm_feat_train_list), y_train_list_baseline)
    high_model = LogisticRegression().fit(np.vstack(high_feat_train_list), y_train_list_highlights)

    print(len(X_test_reg_features))

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, reg_model.predict(X_test_reg_features))
    n_auc = auc(false_positive_rate, true_positive_rate)
    norm_auc.append( n_auc )

    false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test, high_model.predict(X_test_high_features))
    h_auc = auc(false_positive_rate, true_positive_rate)
    high_auc.append( h_auc )

    # find the next most valuable sample to label
    predictions = reg_model.predict(X_train_reg_features)
    baseline_queue_index = find_nearest(predictions,0.5)

    predictions = high_model.predict(X_train_high_features)
    highlight_queue_index = find_nearest(predictions,0.5)


            
with open('norm_auc.pkl', 'wb') as f:
    pickle.dump(norm_auc, f)
with open('high_auc.pkl', 'wb') as f:
    pickle.dump(high_auc, f)
