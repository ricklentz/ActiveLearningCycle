
import pandas as pd
import random
import numpy as np
import pickle

# load data
pos_df = pd.read_csv('../../../Downloads/Batch_3494319_batch_results.csv')
neg_df = pd.read_csv('../../../Downloads/Batch_3494476_batch_results.csv')

# only get the accepted records from the mechanical turk results
pos_df_approved = pos_df[pos_df['AssignmentStatus']=='Approved']
neg_df_approved = neg_df[neg_df['AssignmentStatus']=='Approved']

# get a balanced set of highlighted postive and negative reviews
pos_df_approved = pos_df_approved.drop_duplicates(subset='Input.text', keep='first')
neg_df_approved = neg_df_approved.drop_duplicates(subset='Input.text', keep='first')
balanced_count = min(len(pos_df_approved), len(neg_df_approved))

# buid the dataset to contain the 
labeled_pos = random.sample(pos_df_approved['Answer.highlights'].tolist(), balanced_count)
labeled_neg = random.sample(neg_df_approved['Answer.highlights'].tolist(), balanced_count)

# here, negative is mapped to zere and positive to one
X = (labeled_neg + labeled_pos)
Y = np.concatenate([np.zeros(balanced_count),np.ones(balanced_count)])

# lets save these for the next stage in this pipeline
with open('X.pkl', 'wb') as f:
    pickle.dump(X, f)
with open('Y.pkl', 'wb') as f:
    pickle.dump(Y, f)
