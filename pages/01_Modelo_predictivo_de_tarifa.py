import pandas as pd
import streamlit as st
import pickle
import numpy as np

model_dis = pickle.load(open("./assets/pkl_dis.pkl", 'rb'))
model_dur = pickle.load(open('./assets/pkl_dur.pkl', 'rb'))
model_rate = pickle.load(open('./assets/pkl_rate.pkl', 'rb'))
model_tolls = pickle.load(open('./assets/pkl_tolls.pkl', 'rb'))
model_air = pickle.load(open('./assets/pkl_air.pkl', 'rb'))
model_lin = pickle.load( open('./assets/pkl_lin.pkl', 'rb'))

df = pd.read_parquet('./assets/taxi_trip.parquet', 'pyarrow')
zones_lookup = pd.read_parquet('./assets/zones.parquet', 'pyarrow')
data = {zones_lookup.Zone[i] : zones_lookup.LocationID[i] for i in range(len(zones_lookup.LocationID))}
hour_format = ['0:00 - 1:00','1:00 - 2:00','2:00 - 3:00','3:00 - 4:00','4:00 - 5:00','5:00 - 6:00','6:00 - 7:00','7:00 - 8:00','8:00 - 9:00','9:00 - 10:00','10:00 - 11:00','11:00 - 12:00','12:00 - 13:00','13:00 - 14:00','14:00 - 15:00','15:00 - 16:00','16:00 - 17:00','17:00 - 18:00','18:00 - 19:00','19:00 - 20:00','20:00 - 21:00','21:00 - 22:00','22:00 - 23:00','23:00 - 0:00']
hour = list(range(0, 24))
data_hour = {hour_format[i] : hour[i] for i in range(len(hour_format))}

def fare(est_fare):
     return '{:.2f} USD'.format(est_fare)

def main():
    st.title('Tarifa Estimada')

    option = list(zones_lookup.Zone)
    form = st.form(key='my_form')
    pu_location = form.selectbox('Origen', option)
    do_location = form.selectbox('Destino', option)
    trip_hour = form.selectbox('Hora del viaje', hour_format)
    val = np.array([data[pu_location], data[do_location], data_hour[trip_hour]])
    submit = form.form_submit_button('Mostrar')

    if submit:    
        try:
            val_def = np.array([
                df[(df['PULocationID'] == val[0]) & (df['DOLocationID'] == val[1])]['Trip_Distance'].mean(),
                df[(df['PULocationID'] == val[0]) & (df['DOLocationID'] == val[1])]['Duration_Secs'].mean(),
                df[(df['PULocationID'] == val[0]) & (df['DOLocationID'] == val[1])]['RatecodeID_ml'].median(),
                df[(df['PULocationID'] == val[0]) & (df['DOLocationID'] == val[1])]['Tolls_Amount'].median(),
                df[(df['PULocationID'] == val[0]) & (df['DOLocationID'] == val[1])]['Airport_Fee_ml'].median()
            ])
            st.success(fare(round(model_lin.predict(val_def.reshape(1,-1))[0], 2)))

        except:
            val_def = np.array([
                model_dis.predict(val.reshape(1,-1))[0],
                model_dur.predict(val.reshape(1,-1))[0],
                model_rate.predict(val.reshape(1,-1))[0],
                model_tolls.predict(val.reshape(1,-1))[0],
                model_air.predict(val.reshape(1,-1))[0]
            ])
            st.success(fare(round(model_lin.predict(val_def.reshape(1,-1))[0], 2)))

with st.expander('Contexto'):
    st.write('Modelo general')
    st.image(r"C:\Users\user\Desktop\daniel\DATA SCIENCE\HENRY\HENRY BOOTCAMP\GRUPAL\xd\pages\General.jpg")
    st.write('Modelos generadores de parámetros')
    st.image(r"C:\Users\user\Desktop\daniel\DATA SCIENCE\HENRY\HENRY BOOTCAMP\GRUPAL\xd\pages\Condicionales.jpg")

if __name__ == '__main__':
    main()