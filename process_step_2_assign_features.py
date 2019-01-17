

# load input data
import pickle
import numpy as np
X_train = pickle.load( open('X_train.pkl', 'rb') )
Y_train = pickle.load( open('Y_train.pkl', 'rb') )
X_test = pickle.load( open('X_test.pkl', 'rb') )
Y_test = pickle.load( open('Y_test.pkl', 'rb') )

import pymagnitude
pretrained_magnitude = r'/Users/rwl012/Downloads/pretrained/glove.6B.300d.magnitude'
vectors = pymagnitude.Magnitude(pretrained_magnitude)

# setup speciality cleaning
def get_document_features(data_in,highlights=False,c=0.5):
    data_in = data_in.replace('<span class=\"active_text\">', '').replace('</span>', '')
    body = data_in.split(r'\n                                    ')[1].replace('\n', '')
    body.split(' ')
    avg_vec = np.mean(vectors.query(body), axis=(0))
    
    if highlights == True:
        high_text = data_in.split(r'\n                                    ')[0].replace('\n', '')
        high_avg_vec = np.mean(vectors.query(high_text.split(' ')), axis=(0))
        return (avg_vec,  np.multiply(high_avg_vec,c) + np.multiply(1-c,avg_vec) )
    else:
        return avg_vec
        
# preprocess the training set features
X_train_reg_features = list()
X_train_high_features = list()
for idx,text in enumerate(X_train):
    avg_vec, high_avg_vec = get_document_features(text,True,0.5)
    X_train_reg_features.append(avg_vec)
    X_train_high_features.append(high_avg_vec)
    print("train set " + str(idx/len(X_train)))

# we save these precomputed results since they are used repeatedly downstream
with open('X_train_reg_features.pkl', 'wb') as f:
    pickle.dump(X_train_reg_features, f)
with open('X_train_high_features.pkl', 'wb') as f:
    pickle.dump(X_train_high_features, f)


# preprocess the testing set features
X_test_reg_features = list()
X_test_high_features = list()

for idx,text in enumerate(X_test):
    avg_vec, high_avg_vec = get_document_features(text,True,0.5)
    X_test_reg_features.append(avg_vec)
    X_test_high_features.append(high_avg_vec)
    print("test set " + str(idx/len(X_test)))

# we save these precomputed results since they are used repeatedly downstream
with open('X_test_reg_features.pkl', 'wb') as f:
    pickle.dump(X_test_reg_features, f)
with open('X_test_high_features.pkl', 'wb') as f:
    pickle.dump(X_test_high_features, f)