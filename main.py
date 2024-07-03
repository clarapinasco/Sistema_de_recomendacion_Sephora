import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from plotly import colors 
# Cargar datos (asumiendo que el archivo "product_info.csv" se encuentra en el mismo directorio)
data = pd.read_csv("data/product_info.csv")
reviews = pd.read_csv("data/reviews.csv")
data_pca = pd.read_csv("data/df_pca.csv")
df_als = pd.read_csv("data/df_als.csv")

# Establecer título de la página
st.set_page_config(page_title="Sistema de Recomendación Sephora")

# Crear navbar con Streamlit
sidebar = st.sidebar

# Opciones de navegación
nav_options = ["Home", "Visualizando los datos", "Recomendaciones basadas en colaboración", "Recomendaciones basadas en producto"]

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
                 color_discrete_sequence=['#FFB6C1']
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
                     color_discrete_sequence=['#DB7093'] )

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

        fig = px.pie(values=values, names=labels, hole=0.4, color_discrete_sequence=['#FFC0CB', '#DB7093'])

        # Actualiza las trazas para poner los porcentajes en negrita
        fig.update_traces(texttemplate='<b>%{percent:.2%}</b>', textfont_size=20)

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
            color_discrete_sequence=['#FFC0CB', '#DB7093']
    )
        fig.update_traces(texttemplate='<b>%{percent:.2%}</b>', textfont_size=20)

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

    dr = data.query('brand_name == "Dior"')
    ysl = data.query('brand_name == "Yves Saint Laurent"')
    br = data.query('brand_name == "BURBERRY"')
    ch = data.query('brand_name == "CHANEL"')
    vt = data.query('brand_name == "Valentino"')
    dg = data.query('brand_name == "Dolce&Gabbana"')
    vs = data.query('brand_name == "Versace"')
    mj = data.query('brand_name == "Marc Jacobs Fragrances"')
    tf = data.query('brand_name == "TOM FORD"')
    lm = data.query('brand_name == "La Mer"')
    gr = data.query('brand_name == "GUERLAIN"')
    hm = data.query('brand_name == "HERMÈS"')
    gv = data.query('brand_name == "Givenchy"')
    lc = data.query('brand_name == "Lancôme"')

    # Extrae y limpia los precios en USD de cada marca
    chanel = ch['price_usd'].dropna()
    dior = dr['price_usd'].dropna()
    ystl = ysl['price_usd'].dropna()
    burrberry = br['price_usd'].dropna()
    valentino = vt['price_usd'].dropna()
    versace = vs['price_usd'].dropna()
    dgb = dg['price_usd'].dropna()
    mjb = mj['price_usd'].dropna()
    tomford = tf['price_usd'].dropna()
    guerlain = gr['price_usd'].dropna()
    lamer = lm['price_usd'].dropna()
    hermes = hm['price_usd'].dropna()
    givenchy = gv['price_usd'].dropna()
    lancome = lc['price_usd'].dropna()

    # Define las etiquetas para el gráfico
    labels = ['Dior', 'Yves St. L', 'BURBERRY', 'CHANEL', 'Valentino', 'D&G', 'Versace', 'Marc Jacobs', 'TOM FORD', 'La Mer', 'Guerlain', 'HERMÈS', 'Givenchy', 'Lancôme']


    # Crear el gráfico de caja y bigotes
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.boxplot([dior, ystl, burrberry, chanel, valentino, dgb, versace, mjb, tomford, lamer, guerlain, hermes, givenchy, lancome], labels=labels)
    ax.set_title('Luxury Brands Price Comparison')
    ax.set_ylabel('Price (USD)')
    st.pyplot(fig)

    st.write("*La Mer y TOM FORD*: Son las marcas con los precios más altos en promedio. La Mer, en particular, tiene una amplia variación en sus precios, lo que sugiere una gama de productos con diferentes niveles de precios.")
    st.write("*CHANEL y HERMÈS*: También son marcas con precios relativamente altos. CHANEL tiene una menor variación en los precios en comparación con HERMÈS.")
    st.write("*GUERLAIN, BURBERRY, y Dolce&Gabbana*: Tienen precios medios-altos. GUERLAIN muestra algunos valores atípicos (outliers) que pueden ser productos particularmente caros.")
    st.write("*Yves Saint Laurent, Valentino, Marc Jacobs Fragrances, Dior y Versace*: Estas marcas tienden a tener precios moderados. Yves Saint Laurent tiene una mayor dispersión de precios, lo que sugiere una variedad en la gama de productos.")
    st.write("*Lancôme y Givenchy*: Son las marcas con los precios más bajos en comparación con las demás marcas del gráfico. Lancôme tiene una menor dispersión de precios.")
# Pagina 3 = Comparación de modelos

elif nav_selection == 'Recomendaciones basadas en colaboración':
    def model_backstage():
        st.title('Recomendaciones basadas en tus preferencias')
        st.write('Esta sección explica el trabajo realizado sobre los datos y los pasos que se realizaron para construir el modelo.')

        st.subheader('Carga de Datos')
        st.write('Cargamos varios archivos CSV que contenían datos de reseñas y los combinamos en un único DataFrame. Con esto filtramos los autores que tenían al menos cinco reseñas para asegurar datos significativos para reducir el volumen de los datos y quedarnos con los más interesantes.')
        st.subheader("Modelo ALS")
        st.write("Inicializamos una sesión de Spark y calculamos la dispersión del conjunto de datos para comprender la densidad de las calificaciones disponibles, 99.56%. ")
        st.write("Dividimos nuestro dataset en conjuntos de entrenamiento y prueba, creamos un modelo ALS y lo ajustamos con los datos de entrenamiento. ")
        st.write("Evaluamos el modelo utilizando el error cuadrático medio (RMSE) sobre el conjunto de prueba para medir la precisión de las predicciones y nos dió un RMSE: 0.97")
        st.subheader("Generación de Recomendaciones")
        st.write("Generamos recomendaciones para todos los usuarios y transformamos estas recomendaciones en un formato explorable.")
        st.subheader("Enriquecimiento de Recomendaciones")
        st.write("Cargamos datos adicionales de información de productos y unimos estos datos con nuestras recomendaciones para proporcionar detalles adicionales sobre los productos recomendados.")
        
        st.header("Recomendaciones personalizadas:")
        data['product_id'] = data['product_id'].apply(lambda x: x[1:])
        data['product_id'] = data['product_id'].astype('int64')
        opciones = df_als.merge(data[["product_id", "product_name"]], on='product_id', how='left')
        options = ['Elegí un producto'] + opciones['product_name'].unique().tolist()
        with st.form(key='form_als'):
            # Crear los encabezados
            header = st.columns([1, 2])
            header[0].subheader('Opciones')
            header[1].subheader('Rating')

            # Crear las filas
            rows = []
            for i in range(3):
                row = st.columns([1, 2])
                option = row[0].selectbox(f'Selecciona el producto {i+1}', options, key=f'option_{i}')
                rating = row[1].select_slider(f'Rating {i+1}', options=[1, 2, 3, 4, 5], key=f'rating_{i}')
                rows.append((option, rating))

            # Mostrar los resultados seleccionados
            busqueda = pd.DataFrame(columns=[0, 1])
            # Convertir la lista de diccionarios en un DataFrame y concatenar con el DataFrame existente
            busqueda = pd.concat([busqueda, pd.DataFrame(rows)], ignore_index=True)
            busqueda["author_id"] = df_als.author_id.max() +1
            busqueda.columns = ["product_name", "rating", "author_id"]
            

            boton_buscar = st.form_submit_button("Buscar")
        if boton_buscar:
                #df_spark = spark.createDataFrame(busqueda)
                # Hacer predicciones utilizando el modelo ALS ya entrenado
                #predicciones = modelo_als.transform(df_spark)
                # # Mostrar las primeras 10 filas de las predicciones
                #predicciones.show(10)
            st.write("Buscando...")
            st.write(busqueda)
            
        st.image("images\WhatsApp Image 2024-07-03 at 17.17.25.jpeg")
        st.image("images\WhatsApp Image 2024-07-03 at 17.17.13.jpeg")
        

    model_backstage()

elif nav_selection == 'Recomendaciones basadas en producto':
    def encontra_producto():
        st.title('Busca otros productos similares')

        # Descripción del proceso
        st.write("En esta sección, repasaremos el trabajo realizado con los datos y los pasos llevados a cabo para construir el modelo de recomendación basado en las reseñas de productos de Sephora.")
        st.subheader("**Normalización del Texto**")
        st.write("Para cada producto con reseñas, juntamos todas las reseñas y normalizamos los textos para garantizar una mejor calidad de los datos. Esto incluyó la eliminación de palabras vacías (stopwords), la conversión de todos los caracteres a minúsculas, y la eliminación de puntuación y números. Además, se lematizaron las palabras, lo que significa que se redujeron a su forma base para mejorar la consistencia y reducir la variabilidad.")
        st.subheader("**Vectorización TF-IDF**")
        st.write("Utilizamos la técnica de TF-IDF (Term Frequency-Inverse Document Frequency) para transformar el texto normalizado en una matriz de características. Esta técnica permite identificar las palabras más relevantes en las reseñas, ponderando cada palabra según su frecuencia en comparación con su aparición en otros documentos.")
        st.subheader("**Reducción de Dimensionalidad con PCA**")
        st.write(  r"""    Además de las características extraídas de las reseñas, incluimos variables adicionales del producto: "rating", "reviews", "price_usd", 'limited_edition', 'new', 'online_only', 'out_of_stock', 'sephora_exclusive', 'tertiary_category', y "brand_name" (a las dos últimas se les aplicó la técnica de "get dummies"). Para facilitar el análisis y la visualización de los datos vectorizados, aplicamos el Análisis de Componentes Principales (PCA). Esta técnica ayuda a reducir la dimensionalidad de los datos manteniendo la mayor cantidad posible de variabilidad original, lo que facilita la identificación de patrones y relaciones entre los productos.
    """)
        st.subheader("**Sistema de Recomendación por Similitud**")
        st.write("Desarrollamos un sistema de recomendación por similitud que sugiere productos basándose en la vectorización TF-IDF y la similitud del coseno. Esto significa que, al ingresar un producto, el sistema devuelve productos con descripciones de reseñas y características similares, ofreciendo al usuario recomendaciones relevantes y personalizadas.")
        
        opciones = data_pca.merge(data[["product_id", "product_name"]], on='product_id', how='left')
        
        options = ['Elegí un producto'] + opciones['product_name'].tolist()
        
        with st.form(key='my_form'):
            selected_product_name = st.selectbox("Elegí un producto que te haya gustado", options, index=0)
            boton_buscar = st.form_submit_button("Buscar")
        
        if boton_buscar and selected_product_name != 'Elegí un producto':
            # Función para encontrar productos similares basados en el coseno de similitud
            def find_similar_products(product_id, num_similar_products=3):
                product_indices = data_pca.loc[data_pca['product_id'] == product_id].index
                if len(product_indices) == 0:
                    raise IndexError("Product ID not found in data_pca")
                product_index = product_indices[0]
                # Extraer los componentes principales
                principal_components = data_pca.drop(columns=['product_id']).values
                # Calcular la matriz de similitud de coseno
                cosine_sim = cosine_similarity(principal_components, principal_components)
                # Obtener las similitudes del producto dado
                product_similarities = cosine_sim[product_index]
                # Ordenar y obtener los índices de los productos más similares
                similar_products_indices = product_similarities.argsort()[::-1][1:num_similar_products+1]
                # Obtener los productos similares
                similar_products = data_pca.loc[similar_products_indices, ['product_id']]
                return similar_products
            
            try:
                data_selected_item = data[data["product_name"] == selected_product_name]
                if data_selected_item.empty:
                    st.write('Disculpas, no pudimos encontrar ese producto. Por favor ingresa otro.')
                else:
                    selected_id = data_selected_item['product_id'].iloc[0]
                    similar_products = find_similar_products(selected_id, num_similar_products=3)
                    st.write("Producto buscado:")
                    st.write(data[data["product_id"] == selected_id])
                    st.write('Productos similares:')
                    for i, product in similar_products.iterrows():
                        st.write(f'Product ID: {product.product_id}')
                        st.write(f"Product Name: {data[data['product_id'] == product.product_id]['product_name'].iloc[0]}")
                        st.write(f"Product Brand: {data[data['product_id'] == product.product_id]['brand_name'].iloc[0]}")
                        st.write(f"Price: {data[data['product_id'] == product.product_id]['price_usd'].iloc[0]}")
                        st.write(f"Rating: {data[data['product_id'] == product.product_id]['rating'].iloc[0]}")
                        st.write(f"Highlights: {data[data['product_id'] == product.product_id]['highlights'].iloc[0]}")
                        st.write("--------------------------------------")
            except ValueError as e:
                st.write(e)
                st.write('Disculpas, no pudimos encontrar ese ID de producto. Por favor ingresa otro.')
            except IndexError as e:
                st.write(e)
                st.write('Disculpas, no pudimos encontrar ese ID de producto en los datos PCA. Por favor ingresa otro.')

        st.image('images/img_seph.png')
    
    encontra_producto()
