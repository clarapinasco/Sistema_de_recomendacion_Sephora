import pip
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import pickle
from streamlit_option_menu import option_menu



# from pyspark.sql.functions import col, explode
# from pyspark import SparkContext
# from pyspark.sql import SparkSession
# from scipy.sparse import csr_matrix
# from implicit.als import AlternatingLeastSquares       
# from pyspark import SparkConf
# implicit==0.7.0
# pyspark>=3.0.0


#1. Import
data = pd.read_csv("data\df_limpio.csv")


#2. Titulo de pagina
st.header("Sistema de recomendación de restaurantes")

#3. Sidebar

with st.sidebar:
    selected = option_menu(
        menu_title='Menu',
        options=['Home', 'Visualizando los datos', 'Armado del modelo', 'Encontrá tu restaurant'],
    )

# Resto del código
if selected == 'Home':
    st.title("Home")
    st.write("Bienvenido a la página principal.")
elif selected == 'Visualizando los datos':
    st.title("Visualizando los datos")
    st.write("Aquí puedes visualizar los datos.")
elif selected == 'Armado del modelo':
    st.title("Armado del modelo")
    st.write("Aquí puedes armar el modelo.")
elif selected == 'Encontrá tu libro':
    st.title("Encontrá tu libro")
    st.write("Aquí puedes encontrar tu libro.")

