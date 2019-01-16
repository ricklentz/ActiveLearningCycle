
import pandas as pd
import random
import numpy as np
import pymagnitude

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

#load pretrained model
pretrained_magnitude = r'/Users/rwl012/Downloads/GoogleNews-vectors-negative300.magnitude'
vectors = pymagnitude.Magnitude(pretrained_magnitude)

# setup speciality cleaning
def get_document_features(data_in,highlights=False,c=0.5):
    data_in = data_in.replace('<span class=\"active_text\">', '')
    data_in = data_in.replace('</span>', '')
    body = data_in.split(r'\n                                    ')[1]
    
    body = body.replace('\n', '')
    avg_vec = np.mean(vectors.query(body.split(' ')), axis=(0))
    
    if highlights == True:
        high_text = data_in.split(r'\n                                    ')[0]
        high_text = high_text.replace('\n', '')
        #return np.mean(vectors.query(body.split(' ') + high_text.split(' ')), axis=(0))
        high_avg_vec = np.mean(vectors.query(high_text.split(' ')), axis=(0))
        return (avg_vec, high_avg_vec *(c) + (1-c)*avg_vec)
    else:
        return avg_vec
    
# split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# preprocess the test set features
X_test_reg_features = list()
X_test_high_features = list()

for idx,text in enumerate(X_test):
    avg_vec, high_avg_vec = get_document_features(text,True,0.5)
    X_test_reg_features.append(avg_vec)
    X_test_high_features.append(high_avg_vec)

X_train_reg_features = list()
X_train_high_features = list()
for idx,text in enumerate(X_train):
    avg_vec, high_avg_vec = get_document_features(text,True,0.5)
    X_train_reg_features.append(avg_vec)
    X_train_high_features.append(high_avg_vec)
    
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
# we precompute these since they are used repeatedly downstream
with open('X_test_reg_features.pkl', 'wb') as f:
    pickle.dump(X_test_reg_features, f)
with open('X_test_high_features.pkl', 'wb') as f:
    pickle.dump(X_test_high_features, f)
with open('X_train_reg_features.pkl', 'wb') as f:
    pickle.dump(X_train_reg_features, f)
with open('X_train_high_features.pkl', 'wb') as f:
    pickle.dump(X_train_high_features, f)