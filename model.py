import numpy as np
import pandas as pd
import datetime as dt
 
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, Normalizer, OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score
from scipy.sparse import hstack
import json
from pickle import dump,load
import streamlit as st



## Preprocessing the data
def preprocessing(df):
    """This function prerocesses the dataframe like converting dates to datetime 
    values. """
    
##  1)converting objects to date values , we only need the date not the time

    df['order_purchase_date']=pd.to_datetime(df['order_purchase_date'])
    df['order_customer_delivery_date']=pd.to_datetime(df['order_customer_delivery_date'])
    df['order_estimated_delivery_date']=pd.to_datetime(df['order_estimated_delivery_date'])
    
    return df    


##=======================================================================================


## Feature Engineering
def feature_engineering(df):
    """This function does the feature engineering and adds 5 new engineered features
    to the dataframe"""
    
    # Adding column actual_delivery_days which will tell how many days it actually took for the delivery
    df['actual_delivery_days']=(df['order_customer_delivery_date']-df['order_purchase_date']).dt.days
    
    # Adding column estimated_delivery_days which will tell how manys days the delivery was estimated to be
    df['estimated_delivery_days']=(df['order_estimated_delivery_date']-df['order_purchase_date']).dt.days
    
    # Adding column delivery_accuracy which will tell how many days was the delivery late or early from the estimated delivery
    df['delivery_accuracy']=df['estimated_delivery_days']-df['actual_delivery_days']
    
    # Adding column late_delivery which is a binary column and will tell if the delivery was late or not from the estimated time
    df['late_delivery']=np.where(df['delivery_accuracy']<0,1,0)

    ## Importing seller_popularity_scores json file
    with open('Model parameters\seller_popularity_scores.json','rb') as f4:
                seller_popularity_scores=json.load(f4)
    df['seller_popularity']=seller_popularity_scores.get(df['seller_id'].values[0])
   
    ## We will be dropping the columns that are of no use 
    df.drop(['order_purchase_date','order_customer_delivery_date','order_estimated_delivery_date','seller_id'],axis=1)
    
    return df

##===============================================================================

# @st.cache
def vectorizers():

    ## One hot encoders
    customer_state_ohe=load(open('Model parameters\customer_state_ohe.pkl','rb'))
    product_category_name_ohe=load(open('Model parameters\product_category_name_ohe.pkl','rb'))
    payment_type_ohe=load(open('Model parameters\payment_type_ohe.pkl','rb'))
    seller_state_ohe=load(open('Model parameters\seller_state_ohe.pkl','rb'))

    # Standard Scalers
    price_std=load(open('Model parameters\price_std.pkl','rb'))
    freight_value_std=load(open('Model parameters\\freight_value.pkl','rb'))
    payment_value_std=load(open('Model parameters\payment_value.pkl','rb'))
    product_name_length_std=load(open('Model parameters\product_name_length.pkl','rb'))
    product_description_length_std=load(open('Model parameters\product_description_length.pkl','rb'))
    product_photos_qty_std=load(open('Model parameters\product_photos_qty.pkl','rb'))
    payment_installments_std=load(open('Model parameters\payment_installments.pkl','rb'))
    actual_delivery_days_std=load(open('Model parameters\\actual_delivery_days.pkl','rb'))
    estimated_delivery_days_std=load(open('Model parameters\estimated_delivery_days.pkl','rb'))
    delivery_accuracy_std=load(open('Model parameters\delivery_accuracy.pkl','rb'))

    # MinMax Scaler
    seller_popularity_minmax=load(open('Model parameters\seller_popularity.pkl','rb'))




    return customer_state_ohe,product_category_name_ohe,payment_type_ohe,seller_state_ohe,price_std,freight_value_std,payment_value_std,product_name_length_std,product_description_length_std,product_photos_qty_std,payment_installments_std,actual_delivery_days_std,estimated_delivery_days_std,delivery_accuracy_std,seller_popularity_minmax
                


##===============================================================================

def transform_features(X):


    customer_state_ohe,product_category_name_ohe,payment_type_ohe,seller_state_ohe,price_std,freight_value_std,payment_value_std,product_name_length_std,product_description_length_std,product_photos_qty_std,payment_installments_std,actual_delivery_days_std,estimated_delivery_days_std,delivery_accuracy_std,seller_popularity_minmax = vectorizers()


    
    ## Categorical Features
    
    # customer_state
    customer_state_tf=customer_state_ohe.transform(X['customer_state'].values.reshape(-1,1))

     # product_category_name
    product_category_name_tf=product_category_name_ohe.transform(X['product_category_name'].values.reshape(-1,1))
    
    # payment_type
    payment_type_tf=payment_type_ohe.transform(X['payment_type'].values.reshape(-1,1))
  
    # seller_state
    seller_state_tf=seller_state_ohe.transform(X['seller_state'].values.reshape(-1,1))
    

 ##============================================================================== 
        
    ## Numerical Features
    
    # price
    price_tf=price_std.transform(X['price'].values.reshape(-1,1))

    # freight_value
    freight_value_tf=freight_value_std.transform(X['freight_value'].values.reshape(-1,1))

    # payment_value
    payment_value_tf=payment_value_std.transform(X['payment_value'].values.reshape(-1,1))

    # product_name_length
    product_name_length_tf=product_name_length_std.transform(X['product_name_length'].values.reshape(-1,1))

    # product_description_length
    product_description_length_tf=product_description_length_std.transform(X['product_description_length'].values.reshape(-1,1))

    # product_photos_qty
    product_photos_qty_tf=product_photos_qty_std.transform(X['product_photos_qty'].values.reshape(-1,1))
    
    # payment_installments
    payment_installments_tf=payment_installments_std.transform(X['payment_installments'].values.reshape(-1,1))


    # actual_delivery_days
    actual_delivery_days_tf=actual_delivery_days_std.transform(X['actual_delivery_days'].values.reshape(-1,1))

    # estimated_delivery_days
    estimated_delivery_days_tf=estimated_delivery_days_std.transform(X['estimated_delivery_days'].values.reshape(-1,1))

    # delivery accuracy
    delivery_accuracy_tf=delivery_accuracy_std.transform(X['delivery_accuracy'].values.reshape(-1,1))

    # seller_popularity
    seller_popularity_tf=seller_popularity_minmax.transform(X['seller_popularity'].values.reshape(-1,1))
  
    

 ##=============================================================================

    # Merging all the transformed features
    
    X_enc=hstack((customer_state_tf,product_category_name_tf,payment_type_tf,seller_state_tf,price_tf,freight_value_tf,payment_value_tf,product_name_length_tf,product_description_length_tf,product_photos_qty_tf,payment_installments_tf,actual_delivery_days_tf,estimated_delivery_days_tf,delivery_accuracy_tf,seller_popularity_tf,X['late_delivery'].values.reshape(-1,1)))
        


    return X_enc


##=============================================================================

## Final prediction function
def function_1(X):

    
    if type(X)!=pd.core.frame.DataFrame:
        X=pd.DataFrame(X)
    
    # Preprocessing the model
    X_preprocessed=preprocessing(X)
    
    # Feature Engineering
    X_fe=feature_engineering(X_preprocessed)
    
    # Encoding the features
    X_enc=transform_features(X_fe)

    # Importing final model and its parameters
    model=load(open('Model parameters\model_2.pkl','rb'))

    # model prediction
    predicted_output=model.predict(X_enc)
    
    return predicted_output

