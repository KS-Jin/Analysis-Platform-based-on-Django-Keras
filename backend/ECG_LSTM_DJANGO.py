
# coding: utf-8

# In[1]:
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.naive_bayes import GaussianNB

from keras.models import Sequential
from keras.layers import LSTM, Dense,Flatten,Dropout, Embedding
import numpy as np
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.layers import advanced_activations
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from keras.optimizers import RMSprop
from sklearn.preprocessing import StandardScaler # for normalization
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
#get_ipython().magic('matplotlib inline')


# In[2]:

def post(request):
    #ctx ={}
    ctx = ""
    if request.POST:
        ctx = request.POST['csv']
    return ctx

def read_input(ctx):
    print(ctx)
    data = '../../MITBIH_CSV/csv_result-'
    end = '.csv'
    data_csv = data+ctx+end
    df = pd.read_csv(data_csv)
    df_cp = df.copy()
    return df_cp


# In[23]:


def encode_label(df_cp):
    encoder = LabelBinarizer()
    transformed_label = encoder.fit_transform(df_cp['label'])
    transformed_label = np.array(transformed_label)
    return transformed_label


# In[30]:


def normalize(df_cp):
    df_data= df_cp[['id','Pamp','Ramp','Tamp','PR','QRS','QT','RRI','dRRI']]
    values = df_data.values 
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    return scaled


# In[42]:


def models(transformed_label):
    model = Sequential()
    model.add(Embedding(200000, 250, input_length=9))
    model.add(LSTM(250, dropout=0, recurrent_dropout=0.2))
    model.add(Dense(transformed_label.shape[1], activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    return model


# In[43]:


def ml_train_score(df_cp):
	data = df_cp[['id','Pamp','Ramp','Tamp','PR','QRS','QT','RRI','dRRI']]
	label = df_cp[['label']] 
	X_train1, X_test1, y_train1, y_test1 = train_test_split(data, label , test_size=0.2, random_state=40)
	
	gnb = GaussianNB()
	gnb.fit(X_train1,y_train1)
	gnb_score = gnb.score(X_test1,y_test1)
	
	svc = LinearSVC(random_state=0)
	svc.fit(X_train1,y_train1)
	svc_score = svc.score(X_test1,y_test1)
	
	knn = KNeighborsClassifier(n_neighbors=3)
	knn.fit(X_train1,y_train1)
	knn_score = knn.score(X_test1,y_test1)
	
	RF = RandomForestClassifier(n_jobs=2, random_state=0)
	RF.fit(X_train1,y_train1)
	RF_result = RF.predict(X_test1)
	RF_score = RF.score(X_test1,y_test1)
	return gnb_score,svc_score,knn_score,RF_score

def train_score(scaled,transformed_label,model):
    X_train, X_test, y_train, y_test = train_test_split(scaled,transformed_label, test_size=0.2, random_state=40)
    history = model.fit(X_train, y_train,
          epochs = 1,
          batch_size=100,
          validation_data=(X_test, y_test))
    scores = model.evaluate(X_test, y_test)  
    #print()  
    #print("\t[Info] Accuracy of testing data = {:2.1f}%".format(scores[1]*100.0))
    score, acc = model.evaluate(X_test, y_test, batch_size=100)
    #print('Test score:', score)
    #print('Test accuracy:', acc)
    return score,acc




