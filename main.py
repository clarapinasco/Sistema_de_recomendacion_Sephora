import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from plotly import colors 

# Cargar datos (asumiendo que el archivo "product_info.csv" se encuentra en el mismo directorio)
data = pd.read_csv("data/product_info.csv")
reviews = pd.read_csv("data/reviews.csv")

# Establecer título de la página
st.set_page_config(page_title="Sistema de Recomendación Sephora")

# Crear navbar con Streamlit
sidebar = st.sidebar

# Opciones de navegación
nav_options = ["Home", "Visualizando los datos", "Modelo", "Recomendaciones"]

# Crear barra de navegación usando radio buttons (can be changed to selectbox)
nav_selection = sidebar.radio("Navegación", nav_options)



#####################################################################################################################################

# Pagina 1 = Home
if nav_selection == "Home":
    st.title('Recomendación de productos',)
    st.write('*¡Bienvenido a nuestro sitio de recomendaciones personalizadas de Sephora!*')
    st.image('images/labiales.jpeg', width=800)
    st.write('Descubre productos que se adaptan perfectamente a tus gustos y necesidades con nuestro sistema de recomendación avanzado. ')
    

    st.header('Problemática y objetivos')
    st.markdown('¿Cansado de perderte en el mar de productos de belleza? Analizamos tu historial y preferencias para ofrecerte sugerencias personalizadas.')
    st.markdown('Descubre nuevos productos, ahorra tiempo y toma decisiones inteligentes.')
    st.markdown('¡Prepárate para una experiencia de belleza a tu medida!')
    
    st.header('Dataset')
    st.markdown('El conjunto de datos utilizado, fue extraido de [Kaggle](https://www.kaggle.com/datasets/nadyinky/sephora-products-and-skincare-reviews).')

    st.markdown('Para este proyecto se utiliza una base de 8494 productos y 1.094.411 reviews que realizaron usuarios acerca de los mismos.')
    st.markdown("A continuación podemos ver cómo se componen los set de datos utilizados:")
    st.dataframe(data.head())
    st.dataframe(reviews.head())
    

    # Description of the columns for "Products" table
    st.subheader("Descripción de las columnas - Products")
    st.markdown("\n *product_id* : The unique identifier for the product from the site")
    st.markdown("\n *product_name* : The full name of the product")
    st.markdown("\n *brand_id* : The unique identifier for the product brand from the site")
    st.markdown("\n *brand_name* : The full name of the product brand")
    st.markdown("\n *loves_count* : The number of people who have marked this product as a favorite")
    st.markdown("\n *rating* : The average rating of the product based on user reviews")
    st.markdown("\n *reviews* : The number of user reviews for the product")
    st.markdown("\n *size* : The size of the product, which may be in oz, ml, g, packs, or other units depending on the product type")
    st.markdown("\n *variation_type* : The type of variation parameter for the product (e.g. Size, Color)")
    st.markdown("\n *variation_value* : The specific value of the variation parameter for the product (e.g. 100 mL, Golden Sand)")
    st.markdown("\n *variation_desc* : A description of the variation parameter for the product (e.g. tone for fairest skin)")
    st.markdown("\n *ingredients* : A list of ingredients included in the product, for example: [‘Product variation 1:’, ‘Water, Glycerin’, ‘Product variation 2:’, ‘Talc, Mica’] or if no variations [‘Water, Glycerin’]")
    st.markdown("\n *price_usd* : The price of the product in US dollars")
    st.markdown("\n *value_price_usd* : The potential cost savings of the product, presented on the site next to the regular price")
    st.markdown("\n *sale_price_usd* : The sale price of the product in US dollars")
    st.markdown("\n *limited_edition* : Indicates whether the product is a limited edition or not (1-true, 0-false)")
    st.markdown("\n *new* : Indicates whether the product is new or not (1-true, 0-false)")
    st.markdown("\n *online_only* : Indicates whether the product is only sold online or not (1-true, 0-false)")
    st.markdown("\n *out_of_stock* : Indicates whether the product is currently out of stock or not (1 if true, 0 if false)")
    st.markdown("\n *sephora_exclusive* : Indicates whether the product is exclusive to Sephora or not (1 if true, 0 if false)")
    st.markdown("\n *highlights* : A list of tags or features that highlight the product's attributes (e.g. [‘Vegan’, ‘Matte Finish’])")
    st.markdown("\n *primary_category* : First category in the breadcrumb section")
    st.markdown("\n *secondary_category* : Second category in the breadcrumb section")
#####################################################################################################################################


# Pagina 2 = Graficos
elif nav_selection == 'Visualizando los datos':
    st.title('Visualizando los datos')
    st.image('images/IMG_1634-scaled.png', width=800)

    custom_palette = ['#FFC0CB', '#FFB6C1', '#FF69B4', '#FF1493', '#DB7093', '#FFB5C5', '#FFAEB9', '#FF82AB', '#FF34B3']


    # 1
    def create_primary_category_chart(data):
        category_counts = data['primary_category'].value_counts()
        category_data = pd.DataFrame({'Category': category_counts.index, 'Count': category_counts.values})
        fig = px.bar(category_data, 
                 x='Count', 
                 y='Category', 
                 orientation='h', 
                 color='Count',
                 color_continuous_scale=custom_palette,
                    )
        fig.update_layout(xaxis_title='Count', yaxis_title='Category', yaxis={'categoryorder':'total ascending'})
        return fig
    
    st.title('Categorías de productos')
    fig = create_primary_category_chart(data)
    st.plotly_chart(fig)

    #2
    def create_scatter_plot(data):
   
        fig = px.scatter(data, 
                     x='rating', 
                     y='loves_count', 
                     size='loves_count', 
                     color='rating',
                     color_continuous_scale=custom_palette, )

        fig.update_layout(xaxis_title='Rating 0-5', yaxis_title='Loves Count')
        fig.update_layout(legend=dict(title="Rating", x=1.04, y=1, xanchor="left"))

        return fig


    st.title('Producto Favorito y Rating')
    fig = create_scatter_plot(data)
    st.plotly_chart(fig)
        # Calcular la correlación
    correlation = data['rating'].corr(data['loves_count'])
    st.markdown(f"*Correlation between rating and loves_count:* {correlation:.2f}")
    st.markdown('A medida que el rating aumenta, el "Loves Count" también tiende a aumentar. Existe una tendencia ascendente clara: los productos con ratings más altos tienden a ser agregados a favoritos con mayor frecuencia. Esto sugiere una correlación positiva entre el rating de un producto y cuántas veces es agregado a favoritos.')

    #3
    def create_histogram(data):
        fig = px.histogram(data, 
                       x='rating', 
                       nbins=20, 
                       color_discrete_sequence=['#FFB6C1'], 
                            )

        fig.update_layout(xaxis_title='Rating', yaxis_title='Frequency')
        fig.update_traces(marker_line_color='white', marker_line_width=1)
        return fig
    
    st.title('Histograma de Ratings')
    fig = create_histogram(data)
    st.plotly_chart(fig)
    mean_rating = data['rating'].mean()
    st.write(f"Mean Rating: {mean_rating:.2f}")

     #4
    def create_sephora_exclusive_chart(data):

        df_exclusive = data[data['sephora_exclusive'] == True]
        df_not_exclusive = data[data['sephora_exclusive'] == False]

        count_exclusive = df_exclusive.shape[0]
        count_not_exclusive = df_not_exclusive.shape[0]

        total_products = len(data)
        percent_exclusive = (count_exclusive / total_products) * 100
        percent_not_exclusive = (count_not_exclusive / total_products) * 100

 
        values = [percent_exclusive, percent_not_exclusive]

  
        labels = ['Exclusivo de Sephora', 'No exclusivo de Sephora']

  
        fig = px.pie(values=values, names=labels, hole=0.4, color_discrete_sequence=['#FFC0CB', '#D8BFD8']) 
        

        return fig


    st.title('Productos exclusivos de Sephora')
    fig = create_sephora_exclusive_chart(data)
    st.plotly_chart(fig)


 #6

    def create_online_only_chart(data):
        df_online_only = data[data['online_only'] == True]
        df_not_online_only = data[data['online_only'] == False]

        count_online_only = df_online_only.shape[0]
        count_not_online_only = df_not_online_only.shape[0]

        total_products = len(data)
        percent_online_only = (count_online_only / total_products) * 100
        percent_not_online_only = (count_not_online_only / total_products) * 100

        values = [percent_online_only, percent_not_online_only]
        labels = ['Online Only', 'No Online Only']

        fig = px.pie(
            values=values,
            names=labels,
            hole=0.4,
            color_discrete_sequence=['#FFC0CB', '#D8BFD8']
    )

        return fig


    st.title('Productos exclusivamente online')
    fig = create_online_only_chart(data)
    st.plotly_chart(fig)


    def show_online_only_products(data):
        productos_exclusivos_online = data[data['online_only'] == True].head(5)[['product_name', 'brand_name']]
    

        st.write("### Ejemplos de productos exclusivamente online:")
        for index, row in productos_exclusivos_online.iterrows():
            st.write(f"*{row['product_name']}* - {row['brand_name']}")
    show_online_only_products(data)



    def show_top_expensive_products(data):

        top_10_caros = data.nlargest(10, 'price_usd')[['product_name', 'brand_name', 'price_usd']]

    
        st.title('Productos Más Caros')
        st.subheader('Selecciona un producto para ver su precio:')

    
        selected_product = st.selectbox('Productos', top_10_caros['product_name'])

  
        precio_seleccionado = top_10_caros[top_10_caros['product_name'] == selected_product]['price_usd'].values[0]
        st.write(f"El precio de {selected_product} es: ${precio_seleccionado}")

    show_top_expensive_products(data)


    def show_top_cheapar_products(data):

        top_10_baratos = data.nsmallest(10, 'price_usd')[['product_name', 'brand_name', 'price_usd']]

    
        st.title('Productos Más Económicos')
        st.subheader('Selecciona un producto para ver su precio:')

    
        selected_product = st.selectbox('Productos', top_10_baratos['product_name'])

  
        precio_seleccionado = top_10_baratos[top_10_baratos['product_name'] == selected_product]['price_usd'].values[0]
        st.write(f"El precio de {selected_product} es: ${precio_seleccionado}")

    show_top_cheapar_products(data)

#nube de palabras
   
    st.title('Ingredientes más utilizados')
    st.image('images/nube.png')
    
    ingredientes_repetidos = [
    "Aceite de Semilla (Seed Oil)",
    "Óxidos de Hierro (Iron Oxides)",
    "Extracto de Hoja (Leaf Extract)",
    "Dióxido de Titanio (Titanium Dioxide)",
    "Extracto de Fruta (Fruit Extract)",
    "Agua (Aqua Water)",
    "Ci",
    "Caprilil Glicol (Caprylyl Glycol)"
    ]


    for ingrediente in ingredientes_repetidos:
        st.write(f"- {ingrediente}")

    st.title('Marcas de lujo')
# Pagina 3 = Comparación de modelos

elif nav_selection == 'Modelo':
    def model_backstage():
        st.title('Construcción del modeo')
        st.write('Esta sección explica el trabajo realizado sobre los datos y los pasos que se realizaron para construir el modelo.')
        
    
        st.header('1. Preprocesamiento')

        st.subheader('Tratamiento de nulos y valores faltantes')
        st.write('Fue necesario realizar un proceso detallado de limpieza de los datos, ya que el dataset original contenía valores faltantes o incorrectos en columnas importantes para el armado del modelo.')
        st.write('Eliminamos las columnas con más de un 60 porciento de valores faltantes: sale_price_usd, value_price_usd, variation_desc, child_max_price, child_min_price')
        st.write('La columna size estaba escrita en onzas y en ml. Unificamos para que todo estuviera en oz.')
        
    model_backstage()

elif nav_selection == 'Recomendaciones':
    def encontra_producto():
        st.title('Otros productos que podrían interesarte')
        
        options = ['Elegí un producto'] + data['product_name'].tolist()
        selected_product_name = st.selectbox("Elegí un producto que te haya gustado", options, index=0)
        st.image('images/img_seph.png')
    encontra_producto()
