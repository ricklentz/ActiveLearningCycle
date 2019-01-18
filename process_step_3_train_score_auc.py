# open the upstream data
import pickle
#from sklearn.linear_model import LogisticRegression
#from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

# different values of c (.1) 10 times 
for c in [0.5,0.2,0.8,0.4,0.6,0.7,0.3,0.8,0.9,0.1,0.0,1.0]:
    # import the feature data
    Y = pickle.load(  open('Y.pkl', "rb" ) )
    X_reg_features = pickle.load(  open('X_reg_features.pkl', "rb" ) )
    X_high_features = pickle.load(  open('X_high_features.pkl', "rb" ) )

    for idx,f in enumerate(X_high_features):
        X_high_features[idx] =  (X_high_features[idx] * c) + (1-c) * f

    # reshuffle 20 times and take the average
    # for episode in range(1,20):
    episode = 1

    # split the dataset
    X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X_reg_features, Y, test_size=0.2)
    X_train_high, X_test_high, Y_train_high, Y_test_high = train_test_split(X_high_features, Y, test_size=0.2)

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
    baseline_queue_index = np.nonzero(Y_train_reg == 0)[0][0]
    highlight_queue_index = np.nonzero(Y_train_high == 0)[0][0]

    norm_feat_train_list.append(X_train_reg[baseline_queue_index])
    X_train_reg.pop(baseline_queue_index)
    high_feat_train_list.append(X_train_high[highlight_queue_index])
    X_train_high.pop(highlight_queue_index)

    y_train_list_baseline.append(Y_train_reg[baseline_queue_index])
    Y_train_reg = np.delete(Y_train_reg, Y_train_reg[baseline_queue_index])

    y_train_list_highlights.append(Y_train_high[highlight_queue_index])
    Y_train_high = np.delete(Y_train_high,Y_train_high[highlight_queue_index])

    # let the first iteration 
    baseline_queue_index = np.nonzero(Y_train_reg == 1)[0][0]
    highlight_queue_index = np.nonzero(Y_train_high == 1)[0][0]

    # simulated active learning
    for idx in range(0,len(Y_train_reg)):
        # We add it to the simulated processed data, then remove it from the available training sample queue
        norm_feat_train_list.append(X_train_reg[baseline_queue_index])
        X_train_reg.pop(baseline_queue_index)
        high_feat_train_list.append(X_train_high[highlight_queue_index])
        X_train_high.pop(highlight_queue_index)

        y_train_list_baseline.append(Y_train_reg[baseline_queue_index])
        Y_train_reg = np.delete(Y_train_reg, Y_train_reg[baseline_queue_index])

        y_train_list_highlights.append(Y_train_high[highlight_queue_index])
        Y_train_high = np.delete(Y_train_high,Y_train_high[highlight_queue_index])

        # We now have one of each 
        reg_model = LogisticRegression().fit(np.vstack(norm_feat_train_list), y_train_list_baseline)
        high_model = LogisticRegression().fit(np.vstack(high_feat_train_list), y_train_list_highlights)

        reg_probas_ = reg_model.decision_function(X_test_reg)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test_reg, reg_probas_)
        n_auc = auc(false_positive_rate, true_positive_rate)
        norm_auc.append( n_auc )
        
        high_probas_ = high_model.decision_function(X_test_high)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test_high, high_probas_)
        h_auc = auc(false_positive_rate, true_positive_rate)
        high_auc.append( h_auc )

        # we don't need to predict an empty set
        if len(X_train_reg) > 0:
            # find the index of the next most valuable sample to label
            predictions = reg_model.predict(X_train_reg)
            baseline_queue_index = find_nearest(predictions,0.5)
        
        # we don't need to predict an empty set
        if len(X_train_high) > 0:
            predictions = high_model.predict(X_train_high)
            highlight_queue_index = find_nearest(predictions,0.5)

        print('c = ' + str(c) + ' processed ' + str(idx) + ' remaining ' + str(len(Y_train_reg)))

    with open('norm_auc_'+str(c)+'_'+str(episode)+'.pkl', 'wb') as f:
        pickle.dump(norm_auc, f)
    with open('high_auc_'+str(c)+'_'+str(episode)+'.pkl', 'wb') as f:
        pickle.dump(high_auc, f)


# Plan to complete near term demo for https://sda-rocket-dev.amfam.com as a work around for AWS CloudSearch and EKS/Dask services.  (These are items that the cloud engineering team is working during the sprint that starts tomorrow.)

# The demo requirements are:
# Single user, movie reviews demo showing concept generation, word augmentation

# The required changes are:
# Configure the web application to use lambda services in ieh201 vs ieh106

# The instructions request you only accept the HIT if you agree to the following terms and can meet the specified instructions.  "Highlight all of the relevant words and/or phrases that indicate why the review is positive."  Your submission is rejected because it contains 0 highlighted words and/or phrases.

# The instructions request you only accept the HIT if you agree to the following terms and can meet the specified instructions.  "Highlight all of the relevant words and/or phrases that indicate why the review is negative."  Your submission is rejected because it contains 0 highlighted words and/or phrases.

#   for each text response:
#   randomize selection of highlight per labeled example

# Highlighting random, union, intersection


