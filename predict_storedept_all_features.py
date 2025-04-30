import pandas as pd
from pandas import read_csv
from datetime import datetime
import numpy as np
import statistics
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TimeDistributed
import keras.backend as K
import random
import csv


raw = read_csv("sales_train_validation.csv")
train =  read_csv("sales_test_validation.csv")

train_val = raw
train_val = train_val[train_val['cat_id']=='HOBBIES']



def parse(x):
	return datetime.strptime(x, '%Y-%m-%d')
    
calendar = read_csv('calendar.csv',parse_dates=['date'],index_col = 0, date_parser=parse)
#prices = read_csv('sell_prices.csv')
#prices['item_store'] = prices['item_id']+prices['store_id']

calendar_trim = calendar.drop(["weekday"],axis=1)
calendar_train = calendar_trim[0:1913]
calendar_eval = calendar_trim[1913:]
weekday_train = calendar_train['wday']
weekday_eval =  calendar_eval['wday']

"""
month_train = calendar_train["month"]
month_eval = calendar_eval["month"]

year_train = calendar_train['year']
year_eval = calendar_eval['year']
"""

e1 = calendar_trim['event_name_1']
e2 = calendar_trim['event_name_2']
e1_t = calendar_train['event_type_1']
e2_t = calendar_train['event_type_2']

LE = LabelEncoder()
LE.fit(e1)
e1 = LE.transform(e1)

LE = LabelEncoder()
LE.fit(e2)
e2 = LE.transform(e2)

LE = LabelEncoder()
LE.fit(e1_t)
e1_t = LE.transform(e1_t)

LE = LabelEncoder()
LE.fit(e2_t)
e2_t = LE.transform(e2_t)

e1_train = e1[0:1913]
e1_eval =  e1[1913:]

e2_train = e2[0:1913]
e2_eval =  e2[1913:]

e1_t_train = e1[0:1913]
e1_t_eval =  e1[1913:]

e2_t_train = e2[0:1913]
e2_t_eval =  e2[1913:]

Hobbies_1 = train_val[train_val['dept_id']=='HOBBIES_1']
Hobbies_2 = train_val[train_val['dept_id']=='HOBBIES_2']
Hobbies_1_sales = []
Hobbies_2_sales = []

for i in range(5,1918):
    Hobbies_1_sales.append(sum(Hobbies_1.iloc[:,i]))
    Hobbies_2_sales.append(sum(Hobbies_2.iloc[:,i]))

H1_sales = np.array(Hobbies_1_sales)
H2_sales = np.array(Hobbies_2_sales)

#def splitSequence(seq,wkd,mnt, yr, e1,e2,e1t,e2t, n_steps):
def splitSequence(seq, wkd,e1t,e2t,e1t,e2t, n_steps):
    
    #Declare X and y as empty list
    X = []
    y = []
    
    for i in range(len(seq)):
        #get the last index
        lastIndex = i + n_steps
        
        #if lastIndex is greater than length of sequence then break
        if lastIndex > len(seq) - 1:
            break
            
        #Create input and output sequence
        seq_X, seq_y = seq[i:lastIndex], seq[lastIndex]
        seq_wkd = np.array(wkd[i:lastIndex])
        #seq_mnt, seq_yr = np.array(mnt[i:lastIndex]), np.array(yr[i:lastIndex])
        seq_e1, seq_e2 = np.array(e1[i:lastIndex]),np.array(e2[i:lastIndex])
        seq_e1t, seq_e2t = np.array(e1t[i:lastIndex]),np.array(e2t[i:lastIndex])

        seq_join = np.stack((seq_X, seq_wkd,seq_e1,seq_e2,seq_e1t,seq_e2t),axis=0)
        """
        seq_join = np.stack((seq_X,seq_wkd,seq_mnt, seq_yr, seq_e1,seq_e2,
                             seq_e1t,seq_e2t),axis=0)
        """
        #append seq_X, seq_y in X and y list
        X.append(seq_join)
        y.append(seq_y)
        pass
    #Convert X and y into numpy array
    X = np.array(X)
    y = np.array(y)

    
    return X,y 
    
    pass



def tilted_loss005(y_true, y_pred):
    q = 0.005
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss025(y_true, y_pred):
    q = 0.025
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss165(y_true, y_pred):
    q = 0.165
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss250(y_true, y_pred):
    q = 0.250
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss500(y_true, y_pred):
    q = 0.500
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss750(y_true, y_pred):
    q = 0.750
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss835(y_true, y_pred):
    q = 0.835
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss975(y_true, y_pred):
    q = 0.975
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

def tilted_loss995(y_true, y_pred):
    q = 0.995
    e = (y_true-y_pred)
    return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)

#def make_predictions(sales,wkd,mnt, yr, events1,events2,e1t,e2t, q):
def make_predictions(sales,wkd, events1,events2, e1t,e2t, q):
    data = np.array(sales)
    n_features = 6
    #n_features =8
    n_steps = 56

    #X, y = splitSequence(data,wkd,mnt,yr,events1,events2,e1t,e2t, n_steps)
    X, y = splitSequence(data,wkd,events1,events2,e1t,e2t,n_steps)
    #X = X.reshape((X.shape[0], X.shape[1], n_features))
    X = X.astype(float)
    y = y.astype(float)
    X = tf.convert_to_tensor(X,dtype=tf.float64)
    y = tf.convert_to_tensor(y,dtype=tf.float64)
        
    model = tf.keras.Sequential()
    model.add(layers.LSTM(100, activation='relu', input_shape=(n_features, n_steps)))
    model.add(layers.Dense(1))

    if q == 0.005:
        model.compile(loss=tilted_loss005, optimizer=tf.keras.optimizers.Adam(0.01))
    elif q ==0.025:
        model.compile(loss=tilted_loss025, optimizer=tf.keras.optimizers.Adam(0.01))
    elif q == 0.165:
        model.compile(loss=tilted_loss165, optimizer=tf.keras.optimizers.Adam(0.01))
    elif q ==0.250:
        model.compile(loss=tilted_loss250, optimizer=tf.keras.optimizers.Adam(0.01))
    elif q == 0.500:
        model.compile(loss=tilted_loss500, optimizer=tf.keras.optimizers.Adam(0.01))
    elif q == 0.750:
        model.compile(loss=tilted_loss750, optimizer=tf.keras.optimizers.Adam(0.01))
    elif q == 0.835:
        model.compile(loss=tilted_loss835, optimizer=tf.keras.optimizers.Adam(0.01))
    elif q ==0.975:
        model.compile(loss=tilted_loss975, optimizer=tf.keras.optimizers.Adam(0.01))
    else:
        model.compile(loss=tilted_loss995, optimizer=tf.keras.optimizers.Adam(0.01))
    model.fit(X, y, epochs=10,batch_size = 30,verbose=0)

    data_tail = data[-n_steps:].astype(float)
    wkd_tail = wkd[-n_steps:]
    #month_tail = mnt[-n_steps:]
    #year_tail =yr[-n_steps:]
    e1_tail = events1[-n_steps:]
    e2_tail = events2[-n_steps:]
    e1t_tail = e1t[-n_steps:]
    e2t_tail = e2t[-n_steps:]
    """
    tail = np.stack((data_tail,wkd_tail,month_tail,year_tail,
                     e1_tail,e2_tail, e1t_tail, e2t_tail),axis=0)
    """
    tail = np.stack((data_tail,wkd_tail,e1t_tail, e2t_tail),axis=0)
    tail = tf.convert_to_tensor(tail.reshape((1, n_features, n_steps)),dtype=tf.float64)
    
    prediction = []
    for i in range(28):
        next_num = round(model.predict(tail, verbose=0)[0][0])
        tmp_y = np.append(tail[0][0][1:n_steps].numpy(),next_num)
        tmp_wkd = np.append(tail[0][1][1:n_steps].numpy(), weekday_eval[i])
        #tmp_mnt = np.append(tail[0][2][1:n_steps].numpy(), month_eval[i])
        #tmp_yr = np.append(tail[0][3][1:n_steps].numpy(), year_eval[i])
        tmp_e1 = np.append(tail[0][2][1:n_steps].numpy(), e1_eval[i])
        tmp_e2 = np.append(tail[0][3][1:n_steps].numpy(), e2_eval[i])
        tmp_e1t = np.append(tail[0][4][1:n_steps].numpy(), e1_t_eval[i])
        tmp_e2t = np.append(tail[0][5][1:n_steps].numpy(), e2_t_eval[i])
        tmp = np.stack((tmp_y,tmp_wkd,tmp_e1,tmp_e2, tmp_e1t,tmp_e2t),axis=0)
        tail = tf.convert_to_tensor(tmp.reshape((1,n_features,n_steps)),dtype=tf.float64)
        prediction.append(next_num)

    final_prediction = []
    for pred in prediction:
        if pred <0:
            final_prediction.append(0)
        else:
            final_prediction.append(pred)
        

    return final_prediction

quantiles = (0.005,0.025, 0.165, 0.250, 0.500, 0.750, 0.835, 0.975,0.995)
#quantiles = tf.convert_to_tensor(np.array(quantiles).astype(float))

#H1_eval_sales_28 is y_true    
def SPL(y_true, y_pred, quantile, denom):
    score = 0
    for i in range(len(y_true)):
        yt, yp = y_true[i],y_pred[i]
        if (yp<= yt):
            score = score + ((yt-yp)*quantile)
        else:
            score = (score + ((yp-yt)*(1-quantile)))
    return (score/(28*denom))


train_eval = read_csv("sales_test_validation.csv")
train_eval = train_eval[train_eval['cat_id']=='HOBBIES']

H1_eval = train_eval[train_eval['dept_id']=='HOBBIES_1']
H2_eval = train_eval[train_eval['dept_id']=='HOBBIES_2']
H1_eval_sales = []
H2_eval_sales = []

for i in range(5,len(train_eval.iloc[0,:])):
    H1_eval_sales.append(sum(train_eval.iloc[:,i]))
    H2_eval_sales.append(sum(train_eval.iloc[:,i]))

H1_eval_sales_28 = np.array(H1_eval_sales)
H2_eval_sales_28 = np.array(H2_eval_sales)

#Use H1_sales to get denom
def get_denominator(sales_data):
    diff = 0
    for i in range(1,len(sales_data)):
        diff += abs(sales_data[i] - sales_data[i-1])
    return diff/(len(sales_data)-1)

"""----------------------------Level9-------------------------------"""

Hobbies_1_store_df = {}
Hobbies_2_store_df ={}

H1_store_eval_28 = {}
H2_store_eval_28 = {}

for store in np.unique(Hobbies_1['store_id']):
    df = Hobbies_1[Hobbies_1['store_id']==store]
    df2 = Hobbies_2[Hobbies_2['store_id']==store]
    
    df_eval = H1_eval[H1_eval['store_id']==store]
    df2_eval = H2_eval[H2_eval['store_id']==store]
    
    sales = list()
    sales2 = list()
    
    sales_eval = list()
    sales2_eval = list()
    for i in range(5,1918):
        sales.append(sum(df.iloc[:,i]))
        sales2.append(sum(df2.iloc[:,i]))

    sales, sales2 = np.array(sales), np.array(sales2)
    Hobbies_1_store_df[store] = sales
    Hobbies_2_store_df[store] = sales2

    for j in range(5,33):
        sales_eval.append(sum(df_eval.iloc[:,j]))
        sales2_eval.append(sum(df2_eval.iloc[:,j]))
        
    H1_store_eval_28[store] = np.array(sales_eval)
    H2_store_eval_28[store] = np.array(sales2_eval)

weights = read_csv('weights_evaluation.csv')
weights9 = weights[weights['Level_id']=='Level9']

print('a')
def get_scores(quant):
    score_list = []
    pred_list = []
    for store in H1_store_eval_28.keys():
        """
        preds1 = make_predictions(Hobbies_1_store_df[store], weekday_train, month_train, year_train,
                                  e1_train,e2_train, e1_t_train, e2_t_train, quant)
        """
        
        preds1 = make_predictions(Hobbies_1_store_df[store],weekday_train,e1_train,e2_train,
                                  e1_t_train, e2_t_train, quant)

        preds2 = make_predictions(Hobbies_2_store_df[store],weekday_train,e1_train,e2_train,
                                  e1_t_train, e2_t_train,quant)
        
        SPL1 = SPL(H1_store_eval_28[store],preds1,quant,get_denominator(Hobbies_1_store_df[store]))
        SPL2 = SPL(H2_store_eval_28[store],preds2,quant,get_denominator(Hobbies_2_store_df[store]))

        weight_store = weights9[weights9['Agg_Level_1'] == store]
        w1 = float(weight_store[weight_store['Agg_Level_2'] == 'HOBBIES_1']['weight'])
        w2 = float(weight_store[weight_store['Agg_Level_2'] == 'HOBBIES_2']['weight'])

        score_list.append(['Level9','HOBBIES_1',store,quant,(w1*SPL1/9)])
        score_list.append(['Level9','HOBBIES_2',store,quant,(w2*SPL2/9)])

        line1 = (['Level9', 'HOBBIES_1',store,quant])
        line2 = (['Level9', 'HOBBIES_2',store,quant])

        line1.extend(preds1)
        line2.extend(preds2)
        
        pred_list.append(line1)
        pred_list.append(line2)


    if quant == 0.005:
        print('c')

    return score_list, pred_list
    
csv_results = []
store_predictions = []
rows = []

for q in quantiles:
    print(q)
    temp_result, temp_preds = get_scores(q)
    rows = rows + temp_result
    store_predictions = store_predictions + temp_preds

"""
with open('level9_result_allfeat2.csv','w',newline='') as file:
    writer = csv.writer(file)
    writer.writerows(rows)
   

with open('level9_pred_allfeat2.csv','w',newline='') as file2:
    writer=csv.writer(file2)
    writer.writerows(store_predictions)
"""
WSPL = 0
for r in rows:
        WSPL+= r[4]

                  
#Next Steps
#write to csv scores
#Get percentage per item
#Aggregate per level
#write to csv final


