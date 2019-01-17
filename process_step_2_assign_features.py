
# load input data
import pickle
import numpy as np
X = pickle.load( open('X.pkl', 'rb') )

import pymagnitude
pretrained_magnitude = r'/Users/rwl012/Downloads/pretrained/glove.6B.300d.magnitude'
vectors = pymagnitude.Magnitude(pretrained_magnitude)

# setup speciality cleaning
def get_document_features(data_in,highlights=False,c=0.5):
    data_in = data_in.replace('<span class=\"active_text\">', '').replace('</span>', '')
    body = data_in.split(r'\n                                    ')[1].replace('\n', '')
    avg_vec = np.mean(vectors.query(body.split(' ')), axis=(0))
    
    if highlights == True:
        high_text = data_in.split(r'\n                                    ')[0].replace('\n', '')
        high_avg_vec = np.mean(vectors.query(high_text.split(' ')), axis=(0))
        return avg_vec,  high_avg_vec
    else:
        return avg_vec
        
# preprocess the training set features
X_reg_features = list()
X_high_features = list()
for idx,text in enumerate(X):
    avg_vec, high_avg_vec = get_document_features(text,True,0.5)
    X_reg_features.append(avg_vec)
    X_high_features.append(high_avg_vec)
    print("processing " + str(idx/len(X))) 

# we save these precomputed results since they are used repeatedly downstream
with open('X_reg_features.pkl', 'wb') as f:
     pickle.dump(X_reg_features, f)
with open('X_high_features.pkl', 'wb') as f:
     pickle.dump(X_high_features, f)



