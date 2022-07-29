import numpy as np
import pandas as pd
import streamlit as st
import json
from model import preprocessing, feature_engineering, transform_features, function_1
import json
from pickle import load 
import time



def main():

        st.title('Customer Satisfaction Prediction')
        ### Importing all the necessary parameters
        ## @st.cache
        def starting_param():

                with open('Model parameters\seller_id_list.json','rb') as f1:
                        seller_id_list=json.load(f1)

                with open('Model parameters\state.json','rb') as f2:
                        states=json.load(f2)

                with open('Model parameters\product_category_english.json','rb') as f3:
                        product_category_english=json.load(f3)

                with open('Model parameters\payment_type.json','rb') as f4:
                        payment_type=json.load(f4)

                return states, product_category_english, seller_id_list,payment_type

        states, product_category_english,seller_id_list, payment_type = starting_param()

        order_purchase_date=st.date_input('Date of Order Purchased')
        order_estimated_delivery_date=st.date_input('Estimated date of delivery')
        order_customer_delivery_date=st.date_input('Actual date of delivery')
        customer_state=st.selectbox('State of the customer',states)
        seller_state=st.selectbox('State of the seller',states)
        seller_id=st.selectbox('Seller Id',seller_id_list)
        product_category_name=st.selectbox('Product Category',product_category_english)
        price=st.number_input('Price of the product')
        freight_value=st.number_input('Freight value of the product')
        payment_value=st.number_input('Payment value of the product')
        payment_type=st.selectbox('Select the payment type', payment_type)
        product_name_length=st.number_input('Length of product name')
        product_description_length=st.number_input('Length of product description')
        product_photos_qty=st.slider('Number of photos of product',min_value=0,max_value=10)
        payment_installments=st.slider('Number of Installments',min_value=0,max_value=20)



        if st.button('Click to predict'):

                ## Making the dataframe of all the input values

                df=pd.DataFrame([[order_purchase_date,order_estimated_delivery_date,order_customer_delivery_date,customer_state,seller_state,seller_id,product_category_name,price,freight_value,payment_value,payment_type,product_name_length,product_description_length,product_photos_qty,payment_installments]],columns=['order_purchase_date','order_estimated_delivery_date','order_customer_delivery_date','customer_state','seller_state','seller_id','product_category_name','price','freight_value','payment_value','payment_type','product_name_length','product_description_length','product_photos_qty','payment_installments'])

                with st.spinner('Wait for it....'):
                        time.sleep(2)
                        prediction=function_1(df)

                if prediction[0]==0:
                        st.metric(label="Review", value="Positive")
                elif prediction[0]==1:
                        st.metric(label="Review", value="Negative")
             



if __name__=='__main__':
        main()
