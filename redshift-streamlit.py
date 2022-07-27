import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy import text
import plotly.express as px
import plotly.offline as py
import plotly.figure_factory as ff
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

redshift_endpoint1 = "project-cluster.cztfi4uf8dpc.us-east-1.redshift.amazonaws.com"
redshift_user1 = "alchem1s7"
redshift_pass1 = "fdlr171917X"
port1 = 5439 
dbname1 = "taxitrip"
engine_string = "postgresql+psycopg2://%s:%s@%s:%d/%s" \
% (redshift_user1, redshift_pass1, redshift_endpoint1, port1, dbname1)
engine1 = create_engine(engine_string)

#CREACION DE LAS QUERIES
#Concentracion de viajes por borough
recuento_trips_por_borough = """SELECT COUNT(t.tripid) as numero_de_viajes,
		                                b.borough as borough,
                                        EXTRACT(MONTH FROM t.pu_datetime) as mes
                                FROM taxi_trip t    INNER JOIN zone z ON t.pulocationid = z.locationid 
					                                INNER JOIN borough b ON z.boroughid = b.boroughid
                                WHERE t.activado = 1
                                GROUP BY borough, mes
                                HAVING mes = 3
                                ORDER BY numero_de_viajes DESC;"""
#Viajes con distancias mas largas segun punto de partida
distancias_mas_largas =    """  SELECT  AVG(t.trip_distance) as distancia_promedio,
                                        z.zone as zona
                                FROM taxi_trip t JOIN zone z ON t.pulocationid = z.locationid
                                WHERE t.activado = 1 AND t.distance_outlier = 0
                                GROUP BY zona
                                ORDER BY distancia_promedio DESC
                                LIMIT 15;"""
#Viajes con distancias mas cortas segun punto de partida
distancias_mas_cortas = """     SELECT  AVG(t.trip_distance) as distancia_promedio,
                                        z.zone as zona
                                FROM taxi_trip t JOIN zone z ON t.pulocationid = z.locationid
                                WHERE t.activado = 1
                                GROUP BY zona
                                ORDER BY distancia_promedio ASC 
                                LIMIT 15;"""
#Cantidad promedio de pasajeros por zona de partida
pasajeros_por_pu = """          SELECT  AVG(t.passenger_count) as pasajeros,
                                        z.zone as zona
                                FROM taxi_trip t JOIN zone z ON t.pulocationid = z.locationid
                                WHERE t.activado = 1
                                GROUP BY zona
                                ORDER BY pasajeros DESC 
                                LIMIT 15;  """
#Recorridos mas frecuentes entre origen y destino
most_frequent_trips = """       SELECT  t.pulocationid as localidad_origen,
                                        t.dolocationid as localidad_destino,
                                        COUNT(*) as ocurrence
                                FROM taxi_trip t 
                                GROUP BY localidad_origen, localidad_destino
                                ORDER BY ocurrence DESC
                                LIMIT 15;"""
#Query de la tabla zona para mapear nombres de localidades
zone = """SELECT * FROM zone"""
#Duracion promedio de los viajes en minutos
duration_avg_mins_trips = """   SELECT  AVG(t.duration_secs)/60 as duracion_promedio_minutos, 
                                        z.zone as localidad
                                FROM taxi_trip t INNER JOIN zone z ON t.pulocationid = z.locationid
                                WHERE t.activado = 1 AND t.duration_outlier = 0
                                GROUP BY localidad
                                ORDER BY duracion_promedio_minutos ASC
                                LIMIT 15;"""

#CREACION DE LOS DATAFRAMES
zone_table = pd.read_sql_query(text(zone),engine1)
df_trips_por_borough = pd.read_sql_query(text(recuento_trips_por_borough), engine1)
df_distancias_mas_largas = pd.read_sql_query(text(distancias_mas_largas), engine1)
df_distancias_mas_cortas = pd.read_sql_query(text(distancias_mas_cortas), engine1)
df_pasajeros_por_pu = pd.read_sql_query(text(pasajeros_por_pu), engine1)
df_viajes_mas_frecuentes = pd.read_sql_query(text(most_frequent_trips), engine1)
map_dict = {zone_table.locationid.values[i]:zone_table.zone.values[i] for i in range(0,len(zone_table.zone.values))}
df_viajes_mas_frecuentes["origen"] = df_viajes_mas_frecuentes["localidad_origen"].map(map_dict)
df_viajes_mas_frecuentes["destino"] = df_viajes_mas_frecuentes["localidad_destino"].map(map_dict)
df_viajes_mas_frecuentes["Origen-destino"] = df_viajes_mas_frecuentes["origen"] + "-" + df_viajes_mas_frecuentes["destino"]
df_prom_duracion_trips = pd.read_sql_query(text(duration_avg_mins_trips), engine1)

#CREACION DE LAS TABLAS CON PLOTLY Y DESPLEGUE A STREAMLIT
#Concentracion de viajes por borough
st.title("""Concentracion de viajes dentro de cada borough""")
fig2 = px.bar(df_trips_por_borough, x = "borough", y = "numero_de_viajes", title = "Numero de viajes por borough", labels={"borough":"Borough","numero_de_viajes":"Cantidad de viajes"},color='numero_de_viajes',
            color_continuous_scale=px.colors.sequential.RdBu)
st.plotly_chart(fig2)
with st.expander("See explanation"):
     st.write("""
         Si de numero de recuentos de viajes hablamos, el gráfico arriba mostrado nos indica que Manhattan tiene la mayor concentracion de viajes solicitados.
     """)
#Viajes con distancias mas largas segun punto de partida
st.title("""Zonas donde la distancia promedio de los viajes es mas grande""")
fig3 = px.bar(df_distancias_mas_largas, x = "zona", y = "distancia_promedio", title = "Zonas con distancias mas grandes en promedio", labels={"zona":"Zona de partida","distancia_promedio":"Promedio de distancias (Km)"},color='distancia_promedio',
            color_continuous_scale=px.colors.sequential.Blackbody)
st.plotly_chart(fig3)
with st.expander("See explanation"):
     st.write("""
         En el gráfico se muestran las 15 zonas con viajes mas largos en km, esto para considerarlo al momento de desplegar los servicios de transporte
     """)
#
st.title("""Zonas donde la distancia promedio de los viajes es mas pequeña""")
fig4 = px.bar(df_distancias_mas_cortas, x = "zona", y = "distancia_promedio", title = "Zonas con viajes con distancias mas cortas en promedio", labels={"zona":"Zona de partida","distancia_promedio":"Promedio de distancias (Km) "},color='distancia_promedio',
            color_continuous_scale=px.colors.sequential.Electric)
st.plotly_chart(fig4)
with st.expander("See explanation"):
     st.write("""
        Las 15 zonas con viajes mas cortos nos dicen que los viajes en esta zona generalmente son rapidos, se deben considerar al momento de desplegar los servicios de transporte          
     """)
#
st.title("""Zonas donde los viajes llevan mas pasajeros""")
fig5 = px.bar(df_pasajeros_por_pu, x = "zona", y = "pasajeros", title = "Zonas donde los viajes llevan a mas pasajeros", labels={"zona":"Punto de origen","pasajeros":"Promedion de pasajeros"},color='pasajeros',
            color_continuous_scale=px.colors.sequential.Brwnyl)
st.plotly_chart(fig5)
with st.expander("See explanation"):
     st.write("""
        El estudio de las zonas donde se tienen mas pasajeros nos ayuda a determinar si es necesario desplegar mas unidades de transporte en esa area        
     """)
#
st.title("""Viajes que mas se repiten segun localidad de origen y destino""")
fig6 = px.bar(df_viajes_mas_frecuentes, x = "Origen-destino", y = "ocurrence", title = "Viajes que mas se repiten", labels={"Origen-destino":"Origen-Destino","ocurrence":"Ocurrencia"},color='ocurrence',
            color_continuous_scale=px.colors.sequential.Emrld)
st.plotly_chart(fig6)
with st.expander("See explanation"):
     st.write("""
        El estudio de la ocurrencia de los viajes mas repetidos nos ayuda a concentrar los servicios de transporte en estas areas         
     """)
#
st.title("""Duracion promedio en minutos de los viajes, clasificados segun zona""")
fig7 = px.bar(df_prom_duracion_trips, x = "localidad", y = "duracion_promedio_minutos", title = "Duracion promedio de los viajes segun zona", labels={"duracion_promedio_minutos":"Duracion promedio (min)","localidad":"Zona"},color='duracion_promedio_minutos',
            color_continuous_scale=px.colors.sequential.Blackbody)
st.plotly_chart(fig7)
with st.expander("See explanation"):
     st.write("""
         La duracion de los viajes por zona nos ayuda a corelacionar la distancia con esta variable
     """)
