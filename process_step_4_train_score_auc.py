# open the upstream data
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import numpy as np
import pandas as pd

SEED = 64

def find_nearest(array, value):
    """Finds the prediction closest to the boundary"""
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

# different values of c (.1 increments)
for c in [0.9]:#[0.5,0.2,0.8,0.4,0.6,0.7,0.3,0.9,0.1,0.0,1.0]:
    # import the feature data and labels
    Y = pickle.load(  open('Y.pkl', "rb" ) )
    X_reg_features = pickle.load(  open('X_reg_features.pkl', "rb" ) )
    X_high_features = pickle.load(  open('X_high_features.pkl', "rb" ) )

    # apply c to show how different proportions impact AL performance
    for idx,baseline_feats in enumerate(X_reg_features):
        X_high_features[idx] =  (X_high_features[idx] * c) + (1-c) * baseline_feats

    # reshuffle 20 times and take the average
    for episode in range(2,20):
        #episode = 1

        # split the dataset
        X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X_reg_features, Y, test_size=0.2, random_state=SEED)
        X_train_high, X_test_high, Y_train_high, Y_test_high = train_test_split(X_high_features, Y, test_size=0.2, random_state=SEED)

        # used to accumulate training text samples for simulating active learning
        baseline_train_list = list()
        highlight_train_list = list()

        # used to capture auc for baseline and comparison
        norm_auc = list()
        high_auc = list()

        # seed the initial model with a single instance of each
        baseline_train_list.append( np.nonzero(Y_train_reg == 0)[0][0] )
        baseline_train_list.append( np.nonzero(Y_train_reg == 1)[0][0] )
        highlight_train_list.append( np.nonzero(Y_train_high == 0)[0][0] )
        highlight_train_list.append( np.nonzero(Y_train_high == 1)[0][0] )
      
        # simulated active learning
        for idx in range(0,len(Y_train_reg)):
            # Train each model
            reg_model = LogisticRegression(random_state=SEED).fit( [X_train_reg[i] for i in baseline_train_list],Y_train_reg[baseline_train_list]  )#X_train_reg,Y_train_reg)
            high_model = LogisticRegression(random_state=SEED).fit([X_train_high[i] for i in highlight_train_list]  ,Y_train_high[highlight_train_list] )
            # find AUCs
            n_auc = roc_auc_score(Y_test_reg, reg_model.predict_proba(X_test_reg)[:,1])
            norm_auc.append( n_auc )
            h_auc = roc_auc_score(Y_test_high, high_model.predict_proba(X_test_high)[:,1])
            high_auc.append( h_auc )

            # select most 'unknown' for baseline, but not prediction is needed if empty
            if len(X_train_reg) > 0:
                # find the index of the next most valuable sample to label
                not_selected_reg = set(list(range(len(X_train_reg) )))-set(baseline_train_list)
                predictions = reg_model.predict_proba([X_train_reg[i] for i in not_selected_reg])[:,1]
                baseline_train_list.append(find_nearest(predictions,0.5))
            if len(X_train_high) > 0:
                not_selected_high = set(list(range(len(X_train_high) )))-set(highlight_train_list)
                predictions = high_model.predict_proba([X_train_high[i] for i in not_selected_high])[:,1]
                highlight_train_list.append( find_nearest(predictions,0.5) )

            print('c = ' + str(c) + ' processed ' + str(idx) + ' remaining ' + str(len(Y_train_reg) -len(baseline_train_list) ) + ' auc: ' + str(n_auc) + ' high auc: ' + str(h_auc))

        with open('norm_auc_'+str(c)+'_'+str(episode)+'.pkl', 'wb') as f:
            pickle.dump(norm_auc, f)
        with open('high_auc_'+str(c)+'_'+str(episode)+'.pkl', 'wb') as f:
            pickle.dump(high_auc, f)
