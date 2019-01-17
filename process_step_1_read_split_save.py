
import pandas as pd
import random
import numpy as np

# load data
pos_df = pd.read_csv('/Users/rwl012/Downloads/Batch_3494319_batch_results.csv')
neg_df = pd.read_csv('/Users/rwl012/Downloads/Batch_3494476_batch_results.csv')

# only get those accepted records
pos_df_approved = pos_df[pos_df['AssignmentStatus']=='Approved']
neg_df_approved = neg_df[neg_df['AssignmentStatus']=='Approved']
print(len(pos_df_approved))
print(len(neg_df_approved))

# buid the dataset
labeled_pos = random.sample(pos_df_approved['Answer.highlights'].tolist(), 19000)
labeled_neg = random.sample(neg_df_approved['Answer.highlights'].tolist(), 19000)
Y = np.concatenate([np.zeros(19000),np.ones(19000)])
X = (labeled_neg + labeled_pos)

# split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# and save the inputs
import pickle
with open('X_train.pkl', 'wb') as f:
    pickle.dump(X_train, f)
with open('X_test.pkl', 'wb') as f:
    pickle.dump(X_test, f)
with open('Y_train.pkl', 'wb') as f:
    pickle.dump(Y_train, f)
with open('Y_test.pkl', 'wb') as f:
    pickle.dump(Y_test, f)
