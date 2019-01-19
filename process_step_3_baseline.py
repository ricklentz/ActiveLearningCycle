# open the upstream data
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score
import numpy as np

# import the feature data and labels
Y = pickle.load(  open('Y.pkl', "rb" ) )
X_reg_features = pickle.load(  open('X_reg_features.pkl', "rb" ) )

# split the dataset
X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X_reg_features, Y, test_size=0.2)

# Train each model
reg_model = LogisticRegression().fit(X_train_reg, Y_train_reg)

# find AUC for baseline
reg_probas_ = reg_model.decision_function(X_test_reg)
false_positive_rate, true_positive_rate, thresholds = roc_curve(Y_test_reg, reg_probas_)
print( auc(false_positive_rate, true_positive_rate) ) 