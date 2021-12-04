
def main():
    import streamlit as st

    st.sidebar.header("Selecciona el módulo que quieras implementar:")
    menu = ["Pantalla Principal", "Reglas de Asociación", "Métricas de distancia", "Clustering", "Clasificación (Regresión Logística)", "Árboles de decisión"]
    option = st.sidebar.selectbox("Módulo", menu)

    if option == "Pantalla Principal":
        st.title("SmartsSolutions")
        st.markdown("""
        ### *El aprendizaje automático como una herramienta en la era digital.*
        """)
        imagen = st.image("https://www.muycomputerpro.com/wp-content/uploads/2021/03/inversion-inteligencia-artificial-europa-2021.jpg")
        st.markdown("""
        * Bienvenido a SmartsSolutions, una aplicación desarrollada por *Oscar Casasola*, la cual es una herramienta de apoyo para la implementación de algoritmos de aprendizaje automático.
        * Por favor, selecciona un módulo del menú que se encuentra en el barra lateral izquierda para comenzar...
        """)
        st.markdown("*Solución realizada por: Oscar Casasola.*")

    if option == "Reglas de Asociación":
        from numpy.lib.shape_base import split
        import streamlit as st
        import pandas as pd                 # Para la manipulación y análisis de los datos
        import matplotlib.pyplot as plt     # Para la generación de gráficas a partir de los datos
        import numpy as np                  # Para crear vectores y matrices n dimensionales
        from apyori import apriori         # Para la implementación de reglas de asociación

        st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

        st.title("Módulo: Reglas de asociación")
        st.markdown("""
        * Las **reglas de asociación** es un algoritmo de aprendizaje automático basado en reglas, que se utiliza para encontrar relaciones ocultas en los datos.
        * Se originó con el estudio de transacciones de clientes para determinar asociaciones entre los artículos comprados. También se conoce como **análisis de afinidad**.""")

        datosAsociacion = st.file_uploader("Selecciona un archivo válido para trabajar las reglas de asociación", type=["csv", "txt"])
        
        if datosAsociacion is not None:
            datosRAsociacion = pd.read_csv(datosAsociacion, header=None)
            opcionVisualizacionAsociacion = st.select_slider('Selecciona qué parte de este algoritmo quieres configurar: ', options=["Visualización", "Procesamiento","Implementación del algoritmo"])
            if opcionVisualizacionAsociacion == "Visualización":
                st.header("Datos cargados: ")
                st.dataframe(datosRAsociacion)

            if opcionVisualizacionAsociacion == "Procesamiento":
                Transacciones = datosRAsociacion.values.reshape(-1).tolist() #-1 significa 'dimensión desconocida' (recomendable) o: 7460*20=149200
                ListaM = pd.DataFrame(Transacciones)
                ListaM['Frecuencia'] = 0 #Valor temporal
                #Se agrupa los elementos
                ListaM = ListaM.groupby(by=[0], as_index=False).count().sort_values(by=['Frecuencia'], ascending=True) #Conteo
                ListaM['Porcentaje'] = (ListaM['Frecuencia'] / ListaM['Frecuencia'].sum()) #Porcentaje
                ListaM = ListaM.rename(columns={0 : 'Item'})
                
                column1, column2 = st.columns(2)
                #Se crea una lista con las transacciones
                column1.subheader("Transacciones:")
                column1.dataframe(Transacciones)

                #Se crea una matriz (dataframe) usando la lista y se incluye una columna 'Frecuencia'
                
                column2.subheader("Matriz de transacciones:")
                column2.dataframe(ListaM)

                #Se muestra la lista de las películas menos populares a las más populares
                st.subheader("Elementos de los menos populares a los más populares:")
                st.dataframe(ListaM)

                with st.spinner("Generando gráfica..."):
                    st.subheader("De manera gráfica: ")
                    # Se muestra el gráfico de las películas más populares a las menos populares
                    grafica = plt.figure(figsize=(20,30))
                    plt.xlabel('Frecuencia')
                    plt.ylabel('Elementos')
                    plt.barh(ListaM['Item'], ListaM['Frecuencia'],color='green')
                    plt.title('Elementos de los menos populares a los más populares')
                    st.pyplot(grafica)

            if opcionVisualizacionAsociacion == "Implementación del algoritmo":
                #Se crea una lista de listas a partir del dataframe y se remueven los 'NaN'
                #level=0 especifica desde el primer índice hasta el último
                MoviesLista = datosRAsociacion.stack().groupby(level=0).apply(list).tolist()

                st.subheader("Ingresa los valores deseados para esta configuración del algoritmo: ")
                st.markdown("""
                * **Soporte (Cobertura)**. Indica cuán importante es una regla dentro del total de transacciones.
                * **Confianza**. Indica que tan fiable es una regla.
                * **Lift (Elevación, Interés)**. Indica el nivel de relación (aumento de probabilidad) entre el antecedente y consecuente de la regla. Lift < 1 (Relación negativa), Lift = 1 (Independientes), Lift > 1 (Relación positiva)
                """)
                colu1, colu2, colu3 = st.columns(3)
                min_support =  colu1.number_input("Mínimo de soporte", min_value=0.0, value=0.01, step=0.01)
                min_confidence = colu2.number_input("Mínimo de confianza", min_value=0.0, value=0.3, step=0.01)
                min_lift = colu3.number_input("Mínimo de lift", min_value=0.0, value=2.0, step=1.0)

                if st.checkbox("Mostrar las reglas significativas: "):
                    
                    ReglasC1 = apriori(MoviesLista, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift)
                    Resultado = list(ReglasC1)
                    st.success("Reglas de asociación encontradas: "+ str(len(Resultado)))

                    # Mostrar las reglas de asociación
                    if st.checkbox('Mostrar las reglas de asociación encontradas: '):
                        if len(Resultado) == 0: 
                            st.warning("No se encontraron reglas de asociación")
                        else:
                            c = st.container()
                            col1, col2, col3, col4, col5 = st.columns([1.3,2,1,1,1])
                            with st.container():
                                col1.subheader("Num. regla")
                                col2.subheader("Regla")
                                col3.subheader("Soporte")
                                col4.subheader("Confianza")
                                col5.subheader("Lift")
                                for item in Resultado:
                                    with col1:
                                        #El primer índice de la lista
                                        st.info(str(Resultado.index(item)+1))
                                        Emparejar = item[0]
                                        items = [x for x in Emparejar]
                                    with col2:
                                        #Regla
                                        st.success("("+str(", ".join(item[0]))+")")
                                        
                                    with col3:
                                        # Soporte
                                        st.success(str(round(item[1] * 100,2))+ " %")

                                    with col4:
                                        #Confianza
                                        st.success(str(round(item[2][0][2]*100,2))+ " %")
                                    
                                    with col5:
                                        #Lift
                                        st.success(str(round(item[2][0][3],2))) 
                            
                            # Concluir las reglas de asociación
                            conclusions = st.text_area("En este espacio, se pueden anotar las conclusiones a las que se llegaron a partir de los resultados obtenidos en las reglas de asociación:", "")
                            st.subheader(conclusions)

    if option == "Métricas de distancia":
        import streamlit as st
        import pandas as pd                         # Para la manipulación y análisis de datosMetricas
        import numpy as np                          # Para crear vectores y matrices n dimensionales
        import matplotlib.pyplot as plt             # Para generar gráficas a partir de los datosMetricas
        from scipy.spatial.distance import cdist    # Para el cálculo de distancias
        from scipy.spatial import distance # Para el cálculo de distancias 
        import seaborn as sns


        st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

        st.title("Módulo: Metricas de distancia")
        st.markdown("""
        * Una medida de distancia es una puntuación objetiva que resume la diferencia entre dos elementos (objetos), como: compras, ventas, diagnósticos, personas, usuarios, entre otros.
        * Estas mediciones se utilizan para "aprender de los datos".

        """)
        datosMetricas = st.file_uploader("Selecciona un archivo válido para trabajar con las Métricas de Distancia:", type=["csv","txt"])
        
        if datosMetricas is not None:
            datosMetricasMetricas = pd.read_csv(datosMetricas) 
            st.header("Datos subidos: ")
            st.dataframe(datosMetricasMetricas)

            opcionVisualizacionMetricas = st.select_slider('Selecciona qué métrica de distancia quieres visualizar: ', options=["Euclidiana", "Chebyshev","Manhattan","Minkowski"])
            if opcionVisualizacionMetricas == "Euclidiana":
                st.subheader("Distancia Euclidiana")
                st.markdown("""
                * La **Distancia Euclidiana (euclídea, por Euclides)** es una de las métricas más utilizadas para calcular la distancia entre dos puntos. También es conocida como 'espacio euclidiano'.
                * Sus bases se encuentran en la aplicación del Teorema de Pitágoras, donde la distancia viene a ser la longitud de la hipotenusa. 
                """)
                DstEuclidiana = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='euclidean') # Calcula TODA la matriz de distancias 
                matrizEuclidiana = pd.DataFrame(DstEuclidiana)
                if st.checkbox('Matriz de distancias Euclidiana de todos los objetos'):
                    with st.spinner('Cargando matriz de distancias Euclidiana...'):
                        st.dataframe(matrizEuclidiana)
                        st.subheader("Observando gráficamente la matriz de distancias Euclidiana: ")
                        plt.figure(figsize=(10,10))
                        plt.imshow(matrizEuclidiana, cmap='icefire_r')
                        plt.colorbar()
                        st.pyplot()
                
                if st.checkbox('Distancia Euclidiana entre dos objetos'):
                    with st.spinner('Cargando distancia Euclidiana entre dos objetos...'):
                        #Calculando la distancia entre dos objetos 
                        st.subheader("Selecciona dos objetos para calcular la distancia entre ellos: ")
                        columna1, columna2 = st.columns([1,3])
                        with columna1:
                            objeto1 = st.selectbox('Objeto 1: ', options=matrizEuclidiana.columns)
                            objeto2 = st.selectbox('Objeto 2: ', options=matrizEuclidiana.columns)
                            distanciaEuclidiana = distance.euclidean(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2])
                            st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaEuclidiana))
                        with columna2:
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Euclidiana entre los dos objetos seleccionados")
                            plt.scatter(distanciaEuclidiana, distanciaEuclidiana, c='red',edgecolors='black')
                            plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                            plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                            plt.annotate('  '+str(distanciaEuclidiana.round(2)), xy=(distanciaEuclidiana, distanciaEuclidiana), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaEuclidiana, distanciaEuclidiana))
                            st.pyplot()

                if st.checkbox('Distancia Euclidiana entre dos objetos de tu elección'):
                    with st.spinner('Cargando distancia Euclidiana entre dos objetos de tu elección...'):
                        try:
                            #Calculando la distancia entre dos objetos 
                            st.subheader("Inserta las características de los objetos para calcular la distancia entre ellos: ")
                            columna1, columna2,columna3 = st.columns([1,1,1])
                            with columna1:
                                dimension = st.number_input('Selecciona el número de dimensiones que requieras: ', min_value=1, value=1)
                            
                            objeto1 = []
                            objeto2 = []
                            for p in range(0,dimension):
                                objeto1.append(columna2.number_input('Objeto 1, posición: '+str(p),value=0))
                                objeto2.append(columna3.number_input('Objeto 2, posición: '+str(p),value=0))
                                distanciaEuclidiana = distance.euclidean(objeto1, objeto2)
                                
                            st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaEuclidiana))
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Euclidiana entre los dos objetos seleccionados")
                            plt.scatter(distanciaEuclidiana, distanciaEuclidiana, c='red',edgecolors='black')
                            plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                            plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                            plt.annotate('  '+str(distanciaEuclidiana.round(2)), xy=(distanciaEuclidiana, distanciaEuclidiana), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaEuclidiana, distanciaEuclidiana))
                            st.pyplot()
                        except:
                            st.warning("No se han podido calcular las distancias, intenta con otros valores...")
            
            if opcionVisualizacionMetricas == "Chebyshev":
                st.subheader("Distancia de Chebyshev")
                st.markdown("""
                * La distancia de Chebyshev es el valor máximo absoluto de las diferencias entre las coordenadas de un par de elementos. También es conocida como *"métrica máxima"*.
                * Lleva el nombre del matemático ruso Pafnuty Chebyshev, conocido por su trabajo en la geometría analítica y teoría de números.
                """)
                DstChebyshev = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='chebyshev') # Calcula TODA la matriz de distancias
                matrizChebyshev = pd.DataFrame(DstChebyshev)
                if st.checkbox('Matriz de distancias Chebyshev de todos los objetos'):
                    with st.spinner('Cargando matriz de distancias Chebyshev...'):
                        st.dataframe(matrizChebyshev)
                        st.subheader("Observando gráficamente la matriz de distancias Chebyshev: ")
                        plt.figure(figsize=(10,10))
                        plt.imshow(matrizChebyshev, cmap='icefire_r')
                        plt.colorbar()
                        st.pyplot()

                if st.checkbox('Distancia Chebyshev entre dos objetos'):
                    with st.spinner('Cargando distancia Chebyshev entre dos objetos...'):
                        #Calculando la distancia entre dos objetos 
                        st.subheader("Selecciona dos objetos para calcular la distancia entre ellos: ")
                        columna1, columna2 = st.columns([1,3])
                        with columna1:
                            objeto1 = st.selectbox('Objeto 1: ', options=matrizChebyshev.columns)
                            objeto2 = st.selectbox('Objeto 2: ', options=matrizChebyshev.columns)
                            distanciaChebyshev = distance.chebyshev(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2])
                            st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaChebyshev))
                        with columna2:
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Chebyshev entre los dos objetos seleccionados")
                            plt.scatter(distanciaChebyshev, distanciaChebyshev, c='red',edgecolors='black')
                            plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                            plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                            plt.annotate('  '+str(distanciaChebyshev.round(2)), xy=(distanciaChebyshev, distanciaChebyshev), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaChebyshev, distanciaChebyshev))
                            st.pyplot()

                if st.checkbox('Distancia Chebyshev entre dos objetos de tu elección'):
                    with st.spinner('Cargando distancia Chebyshev entre dos objetos de tu elección...'):
                        try:
                            #Calculando la distancia entre dos objetos 
                            st.subheader("Inserta las características de los objetos para calcular la distancia entre ellos: ")
                            columna1, columna2,columna3 = st.columns([1,1,1])
                            with columna1:
                                dimension = st.number_input('Selecciona el número de dimensiones que requieras: ', min_value=1, value=1)
                            
                            objeto1 = []
                            objeto2 = []
                            for p in range(0,dimension):
                                objeto1.append(columna2.number_input('Objeto 1, posición: '+str(p),value=0))
                                objeto2.append(columna3.number_input('Objeto 2, posición: '+str(p),value=0))
                                distanciaChebyshev = distance.chebyshev(objeto1, objeto2)
                                
                            st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaChebyshev))
                            
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Chebyshev entre los dos objetos seleccionados")
                            plt.scatter(distanciaChebyshev, distanciaChebyshev, c='red',edgecolors='black')
                            plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                            plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                            plt.annotate('  '+str(distanciaChebyshev.round(2)), xy=(distanciaChebyshev, distanciaChebyshev), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaChebyshev, distanciaChebyshev))
                            st.pyplot()
                        except:
                            st.warning("No se han podido calcular las distancias, intenta con otros valores...")

            if opcionVisualizacionMetricas == "Manhattan":
                st.subheader("Distancia de Manhattan")
                st.markdown("""
                * La distancia de Manhattan se utiliza si se necesita calcular la distancia entre dos puntos en una ruta similar a una cuadrícula (información geoespacial).
                * Se llama *Manhattan* debido al diseño de cuadrícula de la mayoría de las calles de la isla de Manhattan, Nueva York (USA).
                """)
                DstManhattan = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='cityblock') # Calcula TODA la matriz de distancias
                matrizManhattan = pd.DataFrame(DstManhattan)
                if st.checkbox('Matriz de distancias Manhattan de todos los objetos'):
                    with st.spinner('Cargando matriz de distancias Manhattan...'):
                        st.dataframe(matrizManhattan)
                        st.subheader("Observando gráficamente la matriz de distancias Manhattan: ")
                        plt.figure(figsize=(10,10))
                        plt.imshow(matrizManhattan, cmap='icefire_r')
                        plt.colorbar()
                        st.pyplot()

                if st.checkbox('Distancia Manhattan entre dos objetos'):
                    with st.spinner('Cargando distancia Manhattan entre dos objetos...'):
                        #Calculando la distancia entre dos objetos 
                        st.subheader("Selecciona dos objetos para calcular la distancia entre ellos: ")
                        columna1, columna2 = st.columns([1,3])
                        with columna1:
                            objeto1 = st.selectbox('Objeto 1: ', options=matrizManhattan.columns)
                            objeto2 = st.selectbox('Objeto 2: ', options=matrizManhattan.columns)
                            distanciaManhattan = distance.cityblock(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2])
                            st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaManhattan))
                        with columna2:
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Manhattan entre los dos objetos seleccionados")
                            plt.scatter(distanciaManhattan, distanciaManhattan, c='red',edgecolors='black')
                            plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                            plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                            plt.annotate('  '+str(distanciaManhattan.round(2)), xy=(distanciaManhattan, distanciaManhattan), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaManhattan, distanciaManhattan))
                            st.pyplot()

                if st.checkbox('Distancia Manhattan entre dos objetos de tu elección'):
                    with st.spinner('Cargando distancia Manhattan entre dos objetos de tu elección...'):
                        try:
                            #Calculando la distancia entre dos objetos 
                            st.subheader("Inserta las características de los objetos para calcular la distancia entre ellos: ")
                            columna1, columna2,columna3 = st.columns([1,1,1])
                            with columna1:
                                dimension = st.number_input('Selecciona el número de dimensiones que requieras: ', min_value=1, value=1)
                            
                            objeto1 = []
                            objeto2 = []
                            for p in range(0,dimension):
                                objeto1.append(columna2.number_input('Objeto 1, posición: '+str(p),value=0))
                                objeto2.append(columna3.number_input('Objeto 2, posición: '+str(p),value=0))
                                distanciaManhattan = distance.cityblock(objeto1, objeto2)
                                
                            st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaManhattan))
                            
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Manhattan entre los dos objetos seleccionados")
                            plt.scatter(distanciaManhattan, distanciaManhattan, c='red',edgecolors='black')
                            plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                            plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                            plt.annotate('  '+str(distanciaManhattan.round(2)), xy=(distanciaManhattan, distanciaManhattan), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaManhattan, distanciaManhattan))
                            st.pyplot()
                        except:
                            st.warning("No se han podido calcular las distancias, intenta con otros valores...")

            if opcionVisualizacionMetricas == "Minkowski":
                st.subheader("Distancia de Minkowski")
                st.markdown("""
                * La distancia de Minkowski es una distancia entre dos puntos en un espacio n-dimensional. 
                * Es una métrica de distancia generalizada: Euclidiana, Manhattan y Chebyshev.
                """)
                DstMinkowski = cdist(datosMetricasMetricas, datosMetricasMetricas, metric='minkowski',p=1.5) # Calcula TODA la matriz de distancias
                matrizMinkowski = pd.DataFrame(DstMinkowski)
                if st.checkbox('Matriz de distancias Minkowski de todos los objetos'):
                    with st.spinner('Cargando matriz de distancias Minkowski...'):
                        st.dataframe(matrizMinkowski)
                        st.subheader("Observando gráficamente la matriz de distancias Minkowski: ")
                        plt.figure(figsize=(10,10))
                        plt.imshow(matrizMinkowski, cmap='icefire_r')
                        plt.colorbar()
                        st.pyplot()

                if st.checkbox('Distancia Minkowski entre dos objetos'):
                    with st.spinner('Cargando distancia Minkowski entre dos objetos...'):
                        #Calculando la distancia entre dos objetos 
                        st.subheader("Selecciona dos objetos para calcular la distancia entre ellos: ")
                        columna1, columna2 = st.columns([1,3])
                        with columna1:
                            objeto1 = st.selectbox('Objeto 1: ', options=matrizMinkowski.columns)
                            objeto2 = st.selectbox('Objeto 2: ', options=matrizMinkowski.columns)
                            distanciaMinkowski = distance.minkowski(datosMetricasMetricas.iloc[objeto1], datosMetricasMetricas.iloc[objeto2], p=1.5)
                            st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaMinkowski))
                        with columna2:
                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Minkowski entre los dos objetos seleccionados")
                            plt.scatter(distanciaMinkowski, distanciaMinkowski, c='red',edgecolors='black')
                            plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                            plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                            plt.annotate('  '+str(distanciaMinkowski.round(2)), xy=(distanciaMinkowski, distanciaMinkowski), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaMinkowski, distanciaMinkowski))
                            st.pyplot()

                if st.checkbox('Distancia Minkowski entre dos objetos de tu elección'):
                    with st.spinner('Cargando distancia Minkowski entre dos objetos de tu elección...'):
                        try:
                            #Calculando la distancia entre dos objetos 
                            st.subheader("Inserta las características de los objetos para calcular la distancia entre ellos: ")
                            columna1, columna2,columna3 = st.columns([1,1,1])
                            with columna1:
                                dimension = st.number_input('Selecciona el número de dimensiones que requieras: ', min_value=1, value=1)

                            objeto1 = []
                            objeto2 = []
                            for p in range(0,dimension):
                                objeto1.append(columna2.number_input('Objeto 1, posición: '+str(p),value=0))
                                objeto2.append(columna3.number_input('Objeto 2, posición: '+str(p),value=0))
                                distanciaMinkowski = distance.minkowski(objeto1, objeto2, p=1.5)
                                
                            st.success("La distancia entre los dos objetos seleccionados es de: "+str(distanciaMinkowski))

                            plt.figure(figsize=(9,9))
                            plt.grid(True)
                            plt.title("Distancia Minkowski entre los dos objetos seleccionados")
                            plt.scatter(distanciaMinkowski, distanciaMinkowski, c='red',edgecolors='black')
                            plt.xlabel('Distancia del objeto '+str(objeto1)+' al objeto '+str(objeto2))
                            plt.ylabel('Distancia del objeto '+str(objeto2)+' al objeto '+str(objeto1))
                            plt.annotate('  '+str(distanciaMinkowski.round(2)), xy=(distanciaMinkowski, distanciaMinkowski), arrowprops=dict(facecolor='red',headwidth=10, headlength=15), xytext=(distanciaMinkowski, distanciaMinkowski))
                            st.pyplot()
                        except:
                            st.warning("No se han podido calcular las distancias, intenta con otros valores...")

    if option == "Clustering":
        from matplotlib import text
        import pandas as pd               # Para la manipulación y análisis de datos
        import numpy as np                # Para crear vectores y matrices n dimensionales
        import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
        import seaborn as sns             # Para la visualización de datos basado en matplotlib
        #%matplotlib inline 
        import streamlit as st            # Para la generación de gráficas interactivas
        from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Para escalar los datos

        #Librerías para el clustering jerárquico 
        import scipy.cluster.hierarchy as shc
        from sklearn.cluster import AgglomerativeClustering
        from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

        #Librerías para Clustering Particional
        from sklearn.cluster import KMeans
        from sklearn.metrics import pairwise_distances_argmin_min
        from kneed import KneeLocator

        st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

        st.title('Módulo: Clustering')
        st.markdown("""
        La IA aplicada en el análisis clústeres consiste en la **segmentación y delimitación de grupos de objetos (elementos), que son unidos por características comunes que éstos comparten** (aprendizaje no supervisado).

        El objetivo es dividir una población heterogénea de elementos en un número de grupos naturales (regiones o segmentos homogéneos), de acuerdo con sus similitudes.
        
        Para hacer clustering es necesario saber el grado de similitud (medidas de distancia) entre los elementos.
        
        """)
        datosCluster = st.file_uploader("Selecciona un archivo válido para trabajar con el módulo de clustering", type=["csv","txt"])
        if datosCluster is not None:
            datosClustering = pd.read_csv(datosCluster)
            datosDelPronostico = []
            for i in range(0, len(datosClustering.columns)):
                datosDelPronostico.append(datosClustering.columns[i])

            opcionClustering1O2 = st.radio("Selecciona el algoritmo de clustering que deseas implementar: ", ('Clustering Jerárquico (Ascendente)', 'Clustering Particional (K-Means)'))

            if opcionClustering1O2 == "Clustering Jerárquico (Ascendente)":
                st.title('Clustering Jerárquico')
                opcionVisualizacionClustersJ = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"])

                if opcionVisualizacionClustersJ == "Evaluación Visual":
                    st.header("Datos cargados: ")
                    st.dataframe(datosClustering)
                    st.subheader("Selecciona la variable a pronosticar: ")
                    variablePronostico = st.selectbox("", datosClustering.columns,index=9)
                    st.write(datosClustering.groupby(variablePronostico).size())
                    try:
                        # Seleccionar los datos que se quieren visualizar
                        st.subheader("Selecciona dos variables que quieras visualizar en el gráfico de dispersión: ")
                        datos = st.multiselect("", datosClustering.columns, default=[datosClustering.columns[4],datosClustering.columns[0]])
                        dato1=datos[0][:]
                        dato2=datos[1][:]

                        with st.spinner("Cargando datos..."):
                            if st.checkbox("Gráfico de dispersión"):
                                sns.scatterplot(x=dato1, y=dato2, data=datosClustering, hue=variablePronostico)
                                plt.title('Gráfico de dispersión')
                                plt.xlabel(dato1)
                                plt.ylabel(dato2)
                                st.pyplot()

                        with st.spinner("Cargando datos..."):
                            if st.checkbox('Ver el gráfico de dispersión de todas las variables con el propósito de seleccionar variables significativas: (puede tardar un poco)'):
                                sns.pairplot(datosClustering, hue=variablePronostico)
                                st.pyplot()
                    except:
                        st.warning("Selecciona solo dos variables...")
                        

                if opcionVisualizacionClustersJ == "Matriz de correlaciones":
                    st.header("MATRIZ DE CORRELACIONES")
                    # MATRIZ DE CORRELACIONES
                    MatrizCorr = datosClustering.corr(method='pearson')
                    st.dataframe(MatrizCorr)
                    #try:
                        #st.subheader("Selecciona una variable para observar cómo se correlaciona con las demás: ")
                        #variableCorrelacion = st.selectbox("", datosClustering.columns) 
                        #st.markdown("**Matriz de correlaciones con la variable seleccionada:** ")
                        #st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores 
                    #except:
                        #st.warning("Selecciona una variable con datos válidos.")

                    # Mapa de calor de la relación que existe entre variables
                    with st.spinner("Cargando mapa de calor..."):
                        st.header("Observando de manera gráfica la matriz de correlaciones: ")
                        plt.figure(figsize=(14,7))
                        MatrizInf = np.triu(MatrizCorr)
                        sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                        plt.title('Mapa de calor de la correlación que existe entre variables')
                        st.pyplot()
                
                if opcionVisualizacionClustersJ == "Aplicación del algoritmo":
                    st.header("Recordando el mapa de calor de la correlación que existe entre variables: ")
                    MatrizCorr = datosClustering.corr(method='pearson')
                    with st.spinner("Cargando mapa de calor..."):
                        plt.figure(figsize=(14,7))
                        MatrizInf = np.triu(MatrizCorr)
                        sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                        plt.title('Mapa de calor de la correlación que existe entre variables')
                        st.pyplot()

                    st.header('Selecciona las variables para hacer el análisis: ')
                    SeleccionVariablesJ = st.multiselect("Selecciona las variables para hacer el análisis: ", datosClustering.columns)
                    MatrizClusteringJ = np.array(datosClustering[SeleccionVariablesJ])
                    if MatrizClusteringJ.size > 0:
                        with st.expander("Da click aquí para visualizar el dataframe de las variables que seleccionaste:"):
                            st.dataframe(MatrizClusteringJ)
                        # Aplicación del algoritmo: 
                        estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
                        MEstandarizada = estandarizar.fit_transform(MatrizClusteringJ)   # Se calculan la media y desviación y se escalan los datos
                        st.subheader("MATRIZ ESTANDARIZADA: ")
                        st.dataframe(MEstandarizada) 

                        st.subheader("Selecciona la métrica de distancias a utilizar: ")
                        metricaElegida = st.selectbox("", ('euclidean','chebyshev','cityblock','minkowski'),index=0)
                        ClusterJerarquico = shc.linkage(MEstandarizada, method='complete', metric=metricaElegida)
                        with st.spinner("Cargando gráfico..."):
                            graficaClusteringJ = plt.figure(figsize=(10, 5))
                            plt.title("Clustering Jerárquico (Ascendente)")
                            plt.xlabel('Observaciones')
                            plt.ylabel('Distancia')
                            Arbol = shc.dendrogram(ClusterJerarquico) #Utilizamos la matriz estandarizada
                            SelectAltura = st.slider('Selecciona a qué nivel quieres "cortar" el árbol: ', min_value=0.0, max_value=np.max(Arbol['dcoord']),step=0.1,value=5.4)
                            plt.axhline(y=SelectAltura, color='black', linestyle='--') # Hace un corte en las ramas
                            st.pyplot(graficaClusteringJ)
                        
                        numClusters = fcluster(ClusterJerarquico, t=SelectAltura, criterion='distance')
                        NumClusters = len(np.unique(numClusters))
                        st.success("El número de clústeres elegido fue de: "+ str(NumClusters))
                        
                        if st.checkbox("Ver los clústeres obtenidos: "):
                            with st.spinner("Cargando..."):
                                try:
                                    #Se crean las etiquetas de los elementos en los clústeres
                                    MJerarquico = AgglomerativeClustering(n_clusters=NumClusters, linkage='complete', affinity=metricaElegida)
                                    MJerarquico.fit_predict(MEstandarizada)
                                    #MJerarquico.labels_

                                    datosClustering = datosClustering[SeleccionVariablesJ]
                                    datosClustering['clusterH'] = MJerarquico.labels_
                                    st.subheader("Dataframe con las etiquetas de los clústeres obtenidos: ")
                                    st.dataframe(datosClustering)

                                    #Cantidad de elementos en los clusters
                                    cantidadElementos = datosClustering.groupby(['clusterH'])['clusterH'].count() 
                                    st.header("Cantidad de elementos en los clústeres: ")
                                    for c in cantidadElementos.index:
                                        st.markdown("En el clúster "+str(c)+" hay **"+str(cantidadElementos[c])+" elementos.**")

                                    # Centroides de los clusters
                                    CentroidesH = datosClustering.groupby('clusterH').mean()
                                    st.header("Centroides de los clústeres: ")
                                    st.table(CentroidesH)

                                    # Interpretación de los clusters
                                    st.header("Interpretación de los clústeres obtenidos: ")
                                    with st.expander("Haz click para visualizar los datos contenidos en cada clúster: "):
                                        for i in range(NumClusters):
                                            st.subheader("Clúster "+str(i))
                                            st.write(datosClustering[datosClustering['clusterH'] == i])
                                    
                                    st.subheader("Interpretación de los centroides de los clústers obtenidos: ")
                                    with st.expander("Haz click para visualizar los centroides obtenidos en cada clúster: "):
                                        for i in range(NumClusters):
                                            st.subheader("Clúster "+str(i))
                                            st.table(CentroidesH.iloc[i])

                                    with st.expander("Haz click para visualizar las conclusiones obtenidas de los centroides de cada clúster: "):
                                        for n in range(NumClusters):
                                            st.subheader("Clúster "+str(n))
                                            st.markdown("**Conformado por: "+str(cantidadElementos[n])+" elementos**")
                                            for m in range(CentroidesH.columns.size):
                                                st.markdown("* Con **"+str(CentroidesH.columns[m])+"** promedio de: "+"**"+str(CentroidesH.iloc[n,m].round(5))+"**.")

                                            st.write("")
                                            st.text_area("Conclusiones del especialista sobre el clúster: "+str(n), " ")
                                        
                                    try: 
                                        # Gráfico de barras de la cantidad de elementos en los clusters
                                        st.header("Representación gráfica de los clústeres obtenidos: ")
                                        plt.figure(figsize=(10, 5))
                                        plt.scatter(MEstandarizada[:,0], MEstandarizada[:,1], c=MJerarquico.labels_)
                                        plt.grid()
                                        st.pyplot()
                                    except:
                                        st.warning("No se pudo graficar.")
                                except: 
                                    st.warning("No se pudo realizar el proceso de clustering, selecciona un 'corte' al árbol que sea correcto")

                    elif MatrizClusteringJ.size == 0:
                        st.warning("No se ha seleccionado ninguna variable.")


            if opcionClustering1O2 == "Clustering Particional (K-Means)":
                st.title('Clustering Particional')

                opcionVisualizacionClustersP = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"])

                if opcionVisualizacionClustersP == "Evaluación Visual":
                    st.header("Datos cargados: ")
                    st.dataframe(datosClustering)
                    st.markdown("**Selecciona la variable a pronosticar:** ")
                    variablePronostico = st.selectbox("", datosClustering.columns,index=9)
                    st.write(datosClustering.groupby(variablePronostico).size())
                    try:
                        # Seleccionar los datos que se quieren visualizar
                        st.subheader("Selecciona dos variables que quieras visualizar en el gráfico de dispersión: ")
                        datos = st.multiselect("", datosClustering.columns, default=[datosClustering.columns[4],datosClustering.columns[0]])
                        dato1=datos[0][:]
                        dato2=datos[1][:]

                        with st.spinner("Cargando datos..."):
                            if st.checkbox("Gráfico de dispersión"):
                                sns.scatterplot(x=dato1, y=dato2, data=datosClustering, hue=variablePronostico)
                                plt.title('Gráfico de dispersión')
                                plt.xlabel(dato1)
                                plt.ylabel(dato2)
                                st.pyplot()

                        with st.spinner("Cargando datos..."):
                            if st.checkbox('Ver el gráfico de dispersión de todas las variables con el propósito de seleccionar variables significativas: (puede tardar un poco)'):
                                sns.pairplot(datosClustering, hue=variablePronostico)
                                st.pyplot()
                    except:
                        st.warning("Selecciona solo dos variables...")


                if opcionVisualizacionClustersP == "Matriz de correlaciones":
                    st.header("MATRIZ DE CORRELACIONES")
                    MatrizCorr = datosClustering.corr(method='pearson')
                    st.dataframe(MatrizCorr)
                    #try:
                        #st.subheader("Selecciona una variable para observar cómo se correlaciona con las demás: ")
                        #variableCorrelacion = st.selectbox("", MatrizCorr.columns) 
                        #st.subheader("Matriz de correlaciones con la variable seleccionada: ")
                        #st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores 
                    #except:
                        #st.warning("Selecciona una variable con datos válidos.")

                    # Mapa de calor de la relación que existe entre variables
                    with st.spinner("Cargando mapa de calor..."):
                        st.header("Observando de manera gráfica la matriz de correlaciones: ")
                        plt.figure(figsize=(14,7))
                        MatrizInf = np.triu(MatrizCorr)
                        sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                        plt.title('Mapa de calor de la correlación que existe entre variables')
                        st.pyplot()
                
                if opcionVisualizacionClustersP == "Aplicación del algoritmo":
                    st.header("Recordando el mapa de calor de la correlación que existe entre variables: ")
                    MatrizCorr = datosClustering.corr(method='pearson')
                    with st.spinner("Cargando mapa de calor..."):
                        plt.figure(figsize=(14,7))
                        MatrizInf = np.triu(MatrizCorr)
                        sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                        plt.title('Mapa de calor de la correlación que existe entre variables')
                        st.pyplot()

                    st.header("Selecciona las variables para hacer el análisis: ")
                    variableSeleccionadas = st.multiselect("", datosClustering.columns)
                    MatrizClusteringP = np.array(datosClustering[variableSeleccionadas])

                    if MatrizClusteringP.size > 0:
                        with st.expander("Da click aquí para visualizar el dataframe de las variables seleccionadas"):
                            st.dataframe(MatrizClusteringP)
                        st.header('Aplicación del algoritmo: K-Means')
                        
                        # Aplicación del algoritmo: 
                        estandarizar = StandardScaler()                               # Se instancia el objeto StandardScaler o MinMaxScaler 
                        MEstandarizada = estandarizar.fit_transform(MatrizClusteringP)   # Se calculan la media y desviación y se escalan los datos
                        st.subheader("MATRIZ ESTANDARIZADA: ")
                        st.dataframe(MEstandarizada) 

                        try: 
                            #Definición de k clusters para K-means
                            #Se utiliza random_state para inicializar el generador interno de números aleatorios
                            k = st.number_input('Selecciona el número de clústeres a implementar: ', min_value=0, value=12, step=1)
                            SSE = []
                            for i in range(2, k):
                                km = KMeans(n_clusters=i, random_state=0)
                                km.fit(MEstandarizada)
                                SSE.append(km.inertia_)
                            
                            #Se grafica SSE en función de k
                            plt.figure(figsize=(10, 7))
                            plt.plot(range(2, k), SSE, marker='o')
                            plt.xlabel('Cantidad de clústeres *k*')
                            plt.ylabel('SSE')
                            plt.title('Elbow Method')
                            st.pyplot()

                            kl = KneeLocator(range(2, k), SSE, curve="convex", direction="decreasing")
                            st.subheader('El codo se encuentra en el clúster número: '+str(kl.elbow))

                            plt.style.use('ggplot')
                            kl.plot_knee()
                            st.pyplot()

                            #Se crean las etiquetas de los elementos en los clústeres
                            MParticional = KMeans(n_clusters=kl.elbow, random_state=0).fit(MEstandarizada)
                            MParticional.predict(MEstandarizada)
                            
                            datosClustering = datosClustering[variableSeleccionadas]
                            datosClustering['clusterP'] = MParticional.labels_
                            st.subheader("Dataframe con las etiquetas de los clústeres obtenidos: ")
                            st.dataframe(datosClustering)

                            #Cantidad de elementos en los clusters
                            numClusters = datosClustering.groupby(['clusterP'])['clusterP'].count() 
                            st.subheader("Cantidad de elementos en los clústeres: ")
                            for i in range(kl.elbow):
                                st.markdown("El clúster número "+str(i)+" tiene **"+str(numClusters[i])+" elementos.**")
                            
                            # Centroides de los clusters
                            CentroidesP = datosClustering.groupby(['clusterP'])[variableSeleccionadas].mean()
                            st.subheader("Centroides de los clústeres: ")
                            st.dataframe(CentroidesP)


                            # Interpretación de los clusters
                            st.header("Interpretación de los clústeres obtenidos: ")
                            with st.expander("Haz click para visualizar los datos contenidos en cada clúster: "):
                                for i in range(kl.elbow):
                                    st.subheader("Clúster "+str(i))
                                    st.write(datosClustering[datosClustering['clusterP'] == i])
                            
                            st.subheader("Interpretación de los centroides de los clústeres obtenidos: ")
                            with st.expander("Haz click para visualizar los centroides obtenidos en cada clúster: "):
                                for i in range(kl.elbow):
                                    st.subheader("Clúster "+str(i))
                                    st.table(CentroidesP.iloc[i])

                            with st.expander("Haz click para visualizar las conclusiones obtenidas de los centroides de cada clúster: "):
                                for n in range(kl.elbow):
                                    st.subheader("Clúster "+str(n))
                                    st.markdown("**Conformado por: "+str(numClusters[n])+" elementos**")
                                    for m in range(CentroidesP.columns.size):
                                        st.markdown("* Con **"+str(CentroidesP.columns[m])+"** promedio de: "+"**"+str(CentroidesP.iloc[n,m].round(5))+"**.")
                                    st.write("")
                                    st.text_area("Conclusiones del especialista sobre el clúster: "+str(n), " ")

                            try: 
                                st.header("Representación gráfica de los clústeres obtenidos: ")
                                from mpl_toolkits.mplot3d import Axes3D
                                plt.rcParams['figure.figsize'] = (10, 7)
                                plt.style.use('ggplot')
                                colores=['red', 'blue', 'green', 'yellow']
                                asignar=[]
                                for row in MParticional.labels_:
                                    asignar.append(colores[row])

                                fig = plt.figure()
                                ax = Axes3D(fig)
                                ax.scatter(MEstandarizada[:, 0], 
                                        MEstandarizada[:, 1], 
                                        MEstandarizada[:, 2], marker='o', c=asignar, s=60)
                                ax.scatter(MParticional.cluster_centers_[:, 0], 
                                        MParticional.cluster_centers_[:, 1], 
                                        MParticional.cluster_centers_[:, 2], marker='o', c=colores, s=1000)
                                st.pyplot()
                            except:
                                st.warning("Ocurrió un error al intentar graficar los clústeres obtenidos...")

                        except:
                            st.warning("Selecciona un número válido de clústeres")
                        

                    elif MatrizClusteringP.size == 0:
                        st.warning("No se ha seleccionado ninguna variable...")

    if option == "Clasificación (Regresión Logística)":
        from typing import BinaryIO
        import pandas as pd               # Para la manipulación y análisis de datos
        import numpy as np                # Para crear vectores y matrices n dimensionales
        import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
        import seaborn as sns             # Para la visualización de datos basado en matplotlib
        #%matplotlib inline 
        import streamlit as st            # Para la generación de gráficas interactivas
        from sklearn.preprocessing import StandardScaler, MinMaxScaler  # Para escalar los datos

        st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

        st.title('Módulo: Regresión Logística')
        st.markdown("""
        * La regresión logística es un de algoritmo de aprendizaje supervisado cuyo objetivo es predecir valores binarios (0 o 1). 
        * Este algoritmo consiste en una transformación a la regresión lineal. 
        * La transformación se debe a que una regresión lineal no funciona para predecir una variable binaria.
        """)

        datosRegresionL = st.file_uploader("Selecciona un archivo válido para trabajar con la regresión logística: ", type=["csv","txt"])
        if datosRegresionL is not None:
            DatosRegresionL = pd.read_csv(datosRegresionL)
            datosDelPronostico = []
            for i in range(0, len(DatosRegresionL.columns)):
                datosDelPronostico.append(DatosRegresionL.columns[i])
            
            opcionVisualizacionRegresionL = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"])

            if opcionVisualizacionRegresionL == "Evaluación Visual":
                st.header("Datos subidos: ")
                st.dataframe(DatosRegresionL)
                st.header("Visualización de los datos")
                variablePronostico = st.selectbox("Variable a clasificar", datosDelPronostico,index=1)
                st.write(DatosRegresionL.groupby(variablePronostico).size())

                # Seleccionar los datos que se quieren visualizar
                try:
                    st.subheader("Gráfico de dispersión")
                    datos = st.multiselect("Selecciona dos variables", datosDelPronostico, default=[datosDelPronostico[0], datosDelPronostico[1]])
                    dato1=datos[0][:]
                    dato2=datos[1][:]

                    if st.checkbox("Visualizar gráfico de dispersión: "):
                        with st.spinner("Cargando gráfico de dispersión..."):
                            sns.scatterplot(x=dato1, y=dato2, data=DatosRegresionL, hue=variablePronostico)
                            plt.title('Gráfico de dispersión')
                            plt.xlabel(dato1)
                            plt.ylabel(dato2)
                            st.pyplot() 
                    
                    if st.checkbox("Visualizar gráfico de dispersión de todas las variables con el fin de seleccionar variables significativas: (puede tardar un poco)"):
                        with st.spinner("Cargando matriz de correlaciones..."):
                            sns.pairplot(DatosRegresionL, hue=variablePronostico)
                            st.pyplot()

                except:
                    st.warning("Selecciona solo dos variables")

            if opcionVisualizacionRegresionL == "Matriz de correlaciones":
                MatrizCorr = DatosRegresionL.corr(method='pearson')
                st.header("Matriz de correlaciones: ")
                st.dataframe(MatrizCorr)

                # SELECCIONAR VARIABLES PARA PRONOSTICAR
                #try:
                    #st.subheader("Correlación de variables: ")
                    #variableCorrelacion = st.selectbox("", MatrizCorr.columns) 
                    #st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores
                #except:
                    #st.warning("Selecciona una variable con datos válidos...")

                # Mapa de calor de la relación que existe entre variables
                st.header("Mapa de calor de la correlación entre variables: ")
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(MatrizCorr)
                sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                plt.title('Mapa de calor de la correlación que existe entre variables')
                st.pyplot()

            if opcionVisualizacionRegresionL == "Aplicación del algoritmo":
                MatrizCorr = DatosRegresionL.corr(method='pearson')
                st.header("Recordando el mapa de calor de la correlación entre variables: ")
                plt.figure(figsize=(14,7))
                MatrizInf = np.triu(MatrizCorr)
                sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                plt.title('Mapa de calor de la correlación que existe entre variables')
                st.pyplot()

                st.header('Definición de variables predictoras (X) y variable clase (Y)')
                st.subheader('Selección de la variable Clase')
                st.markdown('La variable clase debe ser de tipo **DISCRETO**')
                variablePronostico = st.selectbox("Variable a clasificar", DatosRegresionL.columns)

                # Comprobando que la variable clase sea binaria
                if DatosRegresionL[variablePronostico].nunique() == 2:
                    col1, col2, col3 = st.columns(3)
                    # Comprobando el tipo de dato de la variable clase
                    if type(DatosRegresionL[variablePronostico].value_counts().index[1]) and type(DatosRegresionL[variablePronostico].value_counts().index[0]) != np.int64:
                        col1.warning("Para hacer una correcta clasificación, se necesita que los datos a clasificar sean de tipo DISCRETO (0,1)...")
                        col2.error("La etiqueta '"+str(DatosRegresionL[variablePronostico].value_counts().index[1])+"', cambió por el valor 0")
                        col3.success("La etiqueta '"+str(DatosRegresionL[variablePronostico].value_counts().index[0])+"', cambió por el valor 1")
                        with st.expander("Click para ver el dataframe original: "):
                            st.subheader("Dataframe original: ")
                            st.dataframe(DatosRegresionL)
                        with st.expander("Click para ver el dataframe corregido: "):
                            st.subheader("Dataframe corregido: ")
                            DatosRegresionL = DatosRegresionL.replace({str(DatosRegresionL[variablePronostico].value_counts().index[1]): 0, str(DatosRegresionL[variablePronostico].value_counts().index[0]): 1})
                            st.dataframe(DatosRegresionL)
                            Y = np.array(DatosRegresionL[variablePronostico])
                    
                    Y = np.array(DatosRegresionL[variablePronostico])
                    # Variables predictoras
                    st.subheader('Selección de las variables Predictoras')
                    datos = st.multiselect("Selecciona las variables predictoras", DatosRegresionL.columns.drop(variablePronostico))
                    X = np.array(DatosRegresionL[datos])
                    if X.size > 0:
                        with st.expander("Da click aquí para visualizar el dataframe de las variables predictoras que seleccionaste:"):
                            st.dataframe(X)
                        
                        # Seleccionar los datos que se quieren visualizar
                        st.subheader('Visualización de datos: Variables predictoras y su correlación con la variable a clasificar')
                        try:
                            datosPronostico = st.multiselect("Selecciona dos variables: ", datos)
                            datoPronostico1=datosPronostico[0][:]
                            datoPronostico2=datosPronostico[1][:]
                            
                            plt.figure(figsize=(10,7))
                            plt.scatter(X[:,datos.index(datosPronostico[0])], X[:,datos.index(datosPronostico[1])], c=DatosRegresionL[variablePronostico])
                            plt.title('Gráfico de dispersión')
                            plt.xlabel(datoPronostico1)
                            plt.ylabel(datoPronostico2)
                            plt.grid()
                            st.pyplot()
                        except:
                            st.warning("Por favor, selecciona mínimo dos variables que sean válidas para su visualización")
                            datoPronostico1=datosDelPronostico[0][:]
                            datoPronostico2=datosDelPronostico[1][:]
                        
                        
                        try:
                            # Aplicación del algoritmo: Regresión Logística
                            # Se importan las bibliotecas necesarias 
                            from sklearn import linear_model # Para la regresión lineal / pip install scikit-learn
                            from sklearn import model_selection 
                            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

                            st.header('Criterio de división')
                            testSize = st.slider('Selecciona el tamaño del test', min_value=0.2, max_value=0.3, value=0.2, step=0.01)
                            X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=testSize, random_state=1234, shuffle=True)
                            # Datos de entrenamiento: 70, 75 u 80% de los datos
                            # Datos de prueba: 20, 25 o 30% de los datos

                            # DIVISIÓN DE LOS DATOS EN ENTRENAMIENTO Y PRUEBA
                            #st.dataframe(X_train)
                            #st.dataframe(Y_train)

                            # Se entrena el modelo a partir de los datos de entrada
                            Clasificacion = linear_model.LogisticRegression() # Se crea el modelo
                            Clasificacion.fit(X_train, Y_train) # Se entrena el modelo

                            contenedorPredicciones1, contenedorPredicciones2, contenedorPredicciones3 = st.columns(3)
                            with contenedorPredicciones1:
                                # Predicciones probabilísticas
                                st.markdown('Predicciones probabilísticas de los datos de entrenamiento')
                                Probabilidad = Clasificacion.predict_proba(X_train)
                                st.dataframe(Probabilidad)

                            with contenedorPredicciones2:
                                st.markdown('Predicciones probabilísticas de los datos de validación')
                                # Predicciones probabilísticas de los datos de prueba
                                Probabilidad = Clasificacion.predict_proba(X_validation)
                                st.dataframe(Probabilidad) # A partir de las probabilidades se hacen el etiqueta de si es 1 o 0
                            
                            with contenedorPredicciones3:
                                # Predicciones con clasificación final
                                st.markdown('Predicciones con clasificación final')
                                Predicciones = Clasificacion.predict(X_validation)
                                st.dataframe(Predicciones) # A partir de las probabilidades obtenidas anteriormente se hacen las predicciones

                            
                            # Matriz de clasificación
                            st.subheader('Matriz de clasificación')
                            Y_Clasificacion = Clasificacion.predict(X_validation)
                            Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación'])
                            #st.table(Matriz_Clasificacion)
                            
                            col1, col2 = st.columns(2)
                            col1.info('Verdaderos Positivos (VP): '+str(Matriz_Clasificacion.iloc[1,1]))
                            col2.info('Falsos Negativos (FN): '+str(Matriz_Clasificacion.iloc[1,0]))
                            col2.info('Verdaderos Negativos (VN): '+str(Matriz_Clasificacion.iloc[0,0]))
                            col1.info('Falsos Positivos (FP): '+str(Matriz_Clasificacion.iloc[0,1]))

                            # Reporte de clasificación
                            st.subheader('Reporte de clasificación')
                            with st.expander("Da click aquí para ver el reporte de clasificación"):
                                #st.write(classification_report(Y_validation, Y_Clasificacion))
                                st.success("Exactitud promedio de la validación: "+str(Clasificacion.score(X_validation, Y_validation).round(6)*100)+" %")
                                precision = float(classification_report(Y_validation, Y_Clasificacion).split()[10])*100
                                st.success("Precisión: "+ str(precision)+ " %")
                                st.error("Tasa de error: "+str((1-Clasificacion.score(X_validation, Y_validation))*100)+" %")
                                sensibilidad = float(classification_report(Y_validation, Y_Clasificacion).split()[11])*100
                                st.success("Sensibilidad: "+ str(sensibilidad)+ " %")
                                especificidad = float(classification_report(Y_validation, Y_Clasificacion).split()[6])*100
                                st.success("Especificidad: "+ str(especificidad)+" %")
                            
                            st.subheader('Modelo de clasificación: ')
                            # Ecuación del modelo
                            st.latex(r"p=\frac{1}{1+e^{-(a+bX)}}")
                            
                            with st.expander("Da click aquí para ver la ecuación del modelo"):
                                st.success("Intercept: "+str(Clasificacion.intercept_[0]))
                                #st.write("Coeficientes:\n",Clasificacion.coef_)
                                st.markdown("**Ecuación del modelo:** ")
                                st.latex("a+bX="+str(Clasificacion.intercept_[0]))
                                for i in range(len(datos)):
                                    datos[i] = datos[i].replace("_", "")
                                    st.latex("+"+str(Clasificacion.coef_[0][i].round(6))+"("+str(datos[i])+")")
                                
                            st.subheader('Clasificación basada en el modelo establecido')
                            with st.expander("Da click aquí para clasificar los datos que gustes"):
                                st.subheader('Clasificación de casos')
                                sujetoN = st.text_input("Ingrese el nombre o ID del sujeto que desea clasificar: ")

                                dato = []
                                for p in range(len(datos)):
                                    dato.append(st.number_input(datos[p][:], step=0.1))
                                
                                if st.checkbox("Dar clasificación: "):
                                    if Clasificacion.predict([dato])[0] == 0:
                                        st.error("Con un algoritmo que tiene una exactitud del: "+str(round(Clasificacion.score(X_validation, Y_validation)*100,2))+"%, la clasificación para el sujeto "+str(sujetoN)+", tomando en cuenta como variable predictora: '"+str(variablePronostico)+"', fue de 0 (CERO)")
                                    elif Clasificacion.predict([dato])[0] == 1:
                                        st.success("Con un algoritmo que tiene una exactitud del: "+str(round(Clasificacion.score(X_validation, Y_validation)*100,2))+"%, la clasificación para el sujeto "+str(sujetoN)+", tomando en cuenta como variable predictora: '"+str(variablePronostico)+"', fue de 1 (UNO)")
                                    else:
                                        st.warning("El resultado no pudo ser determinado, intenta hacer una buena selección de variables")
                        except:
                            st.warning("No se pudo realizar la clasificación porque no se ha hecho una correcta selección de variables")
                            
                    elif X.size == 0:
                        st.warning("No se han seleccionado variables predictoras...")

                elif DatosRegresionL[variablePronostico].nunique() != 2:
                    st.warning("La variable clase no contiene datos binarios, por lo que no se puede realizar la clasificación... intenta con otra variable")


    if option == "Árboles de decisión":
        import pandas as pd               # Para la manipulación y análisis de datos
        import numpy as np                # Para crear vectores y matrices n dimensionales
        import matplotlib.pyplot as plt   # Para la generación de gráficas a partir de los datos
        import seaborn as sns             # Para la visualización de datos basado en matplotlib
        #%matplotlib inline 
        import streamlit as st            # Para la generación de gráficas interactivas
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        from sklearn import model_selection

        st.set_option('deprecation.showPyplotGlobalUse', False) # Para evitar que se muestre el warning de matplotlib

        st.title('Módulo: Árboles de decisión')
        st.markdown("""
        * Es uno de los algoritmos más utilizados en el aprendizaje automático supervisado.
        * Permiten resolver problemas de regresión (pronóstico) y clasificación.
        * Aportan claridad (despliegan los resultados en profundidad, de mayor a menor detalle).
        * Tienen buena precisión en un amplio número de aplicaciones.
        """)
        datosArboles = st.file_uploader("Selecciona un archivo válido para trabajar con los árboles de decisión: ", type=["csv","txt"])
        if datosArboles is not None:
            datosArbolesDecision = pd.read_csv(datosArboles)
            datosDelPronostico = []
            for i in range(0, len(datosArbolesDecision.columns)):
                datosDelPronostico.append(datosArbolesDecision.columns[i])
            
            opcionArbol1O2 = st.radio("Selecciona el tipo de árbol de decisión que deseas utilizar: ", ("Árbol de decisión (Regresión)", "Árbol de decisión (Clasificación)"))
            
            if opcionArbol1O2 == "Árbol de decisión (Regresión)":

                opcionVisualizacionArbolD = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"], value="Evaluación Visual")

                if opcionVisualizacionArbolD == "Evaluación Visual":
                    st.subheader("Evaluación visual de los datos cargados: ")
                    st.dataframe(datosArbolesDecision)
                    st.subheader("Datos estadísticos: ")
                    st.dataframe(datosArbolesDecision.describe())

                    st.subheader("Gráficamente")
                    variablePronostico = st.selectbox("Selecciona una variable a visualizar", datosArbolesDecision.columns.drop('IDNumber'),index=4)
                    if st.checkbox("Da click para cargar la gráfica (puede tardar un poco)"):
                        with st.spinner('Cargando gráfica...'):
                            plt.figure(figsize=(20, 5))
                            plt.plot(datosArbolesDecision['IDNumber'], datosArbolesDecision[variablePronostico], color='green', marker='o', label=variablePronostico)
                            plt.xlabel('Paciente')
                            plt.ylabel(variablePronostico)
                            plt.title('Pacientes con tumores cancerígenos')
                            plt.grid(True)
                            plt.legend()
                            st.pyplot()
                    
                if opcionVisualizacionArbolD == "Matriz de correlaciones":
                    MatrizCorr = datosArbolesDecision.corr(method='pearson')
                    st.header("Matriz de correlaciones: ")
                    st.dataframe(MatrizCorr)

                    # SELECCIONAR VARIABLES PARA PRONOSTICAR
                    #try:
                        #st.subheader("Correlación de variables: ")
                        #variableCorrelacion = st.selectbox("", MatrizCorr.columns) 
                        #st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores
                    #except:
                        #st.warning("Selecciona una variable con datos válidos...")

                    # Mapa de calor de la relación que existe entre variables
                    st.header("Mapa de calor de la correlación entre variables: ")
                    plt.figure(figsize=(14,7))
                    MatrizInf = np.triu(MatrizCorr)
                    sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                    plt.title('Mapa de calor de la correlación que existe entre variables')
                    st.pyplot()

                if opcionVisualizacionArbolD == "Aplicación del algoritmo":
                    st.header('Definición de variables predictoras (X) y variable clase (Y)')
                    st.subheader("Recordando la matriz de correlaciones: ")
                    MatrizCorr = datosArbolesDecision.corr(method='pearson')
                    st.dataframe(MatrizCorr)

                    st.subheader('Variables Predictoras')
                    datosADeciR = st.multiselect("Datos", datosDelPronostico)
                    X = np.array(datosArbolesDecision[datosADeciR]) 
                    if X.size > 0:
                        with st.expander("Da click aquí para visualizar el dataframe de las variables predictoras que seleccionaste:"):
                            st.dataframe(X)

                        st.subheader('Variable Clase')
                        variablePronostico = st.selectbox("Variable a pronosticar", datosArbolesDecision.columns.drop(datosADeciR),index=3)
                        Y = np.array(datosArbolesDecision[variablePronostico])
                        with st.expander("Da click aquí para visualizar el dataframe con la variable clase que seleccionaste:"):
                            st.dataframe(Y)
                
                        try:
                            # Aplicación del algoritmo: Regresión Logística
                            # Se importan las bibliotecas necesarias 
                            from sklearn import linear_model # Para la regresión lineal / pip install scikit-learn
                            from sklearn import model_selection 
                            from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

                            st.header('Criterio de división')
                            testSize = st.slider('Selecciona el tamaño del test', min_value=0.2, max_value=0.3, value=0.2, step=0.01)
                            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=testSize, random_state=1234, shuffle=True)
                            # Datos de entrenamiento: 70, 75 u 80% de los datos
                            # Datos de prueba: 20, 25 o 30% de los datos

                            # DIVISIÓN DE LOS DATOS EN ENTRENAMIENTO Y PRUEBA
                            #st.dataframe(X_train)
                            #st.dataframe(Y_train)

                            # SE ENTRENA EL MODELO A TRAVÉS DE UN ÁRBOL DE DECISIÓN (REGRESIÓN)
                            # EXPLICACIÓN 
                            st.header('Parámetros del árbol de decisión: ')
                            st.markdown("""
                            * **max_depth**. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.
                            * **min_samples_leaf**. Indica la cantidad mínima de datos que debe tener un nodo hoja.
                            * **min_samples_split**. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.
                            * **criterion**. Indica la función que se utilizará para dividir los datos. Puede ser (ganancia de información) gini y entropy (Clasificación). Cuando el árbol es de regresión se usan funciones como el error cuadrado medio (MSE). """)

                            st.write("Selecciona los valores que requieras para entrenar el modelo: ")
                            choiceProfuncidad = st.select_slider('Máxima profundidad del árbol (max_depth)', options=["None","Valores numéricos"], value="None")
                            column1, column2, column3 = st.columns(3)
                            if choiceProfuncidad == "None":
                                Max_depth = None
                            elif choiceProfuncidad == "Valores numéricos":
                                Max_depth = column1.number_input('Máxima profundidad del árbol (max_depth)', min_value=1, value=8)

                            Min_samples_split = column2.number_input('min_samples_split', min_value=1, value=2)
                            Min_samples_leaf = column3.number_input('min_samples_leaf', min_value=1, value=1)
                            Criterio = st.selectbox('criterion', options=["squared_error", "friedman_mse", "absolute_error", "poisson"])
                            
                            PronosticoAD = DecisionTreeRegressor(max_depth=Max_depth, min_samples_split=Min_samples_split, min_samples_leaf=Min_samples_leaf, criterion=Criterio,random_state=0)
                            #PronosticoAD = DecisionTreeRegressor(max_depth=8, min_samples_split=4, min_samples_leaf=2)
                            
                            PronosticoAD.fit(X_train, Y_train)
                            #Se genera el pronóstico
                            Y_Pronostico = PronosticoAD.predict(X_test)
                            st.subheader('Datos del test vs Datos del pronóstico')
                            Valores = pd.DataFrame(Y_test, Y_Pronostico)
                            st.dataframe(Valores)

                            st.subheader('Gráficamente: ')
                            plt.figure(figsize=(20, 5))
                            plt.plot(Y_test, color='green', marker='o', label='Y_test')
                            plt.plot(Y_Pronostico, color='red', marker='o', label='Y_Pronostico')
                            plt.xlabel('Paciente')
                            plt.ylabel('Tamaño del tumor')
                            plt.title('Pacientes con tumores cancerígenos')
                            plt.grid(True)
                            plt.legend()
                            st.pyplot()

                            # Reporte de clasificación
                            st.subheader('Reporte de clasificación')
                            with st.expander("Da click aquí para ver el reporte de clasificación"):
                                st.success('Criterio: '+str(PronosticoAD.criterion))
                                st.success('Importancia variables: '+str(PronosticoAD.feature_importances_))
                                st.success("MAE: "+str(mean_absolute_error(Y_test, Y_Pronostico)))
                                st.success("MSE: "+str(mean_squared_error(Y_test, Y_Pronostico)))
                                st.success("RMSE: "+str(mean_squared_error(Y_test, Y_Pronostico, squared=False)))   #True devuelve MSE, False devuelve RMSE
                                st.success('Score (exactitud promedio de la validación): '+str(r2_score(Y_test, Y_Pronostico).round(6)*100)+" %")
                            
                            st.subheader('Importancia de las variables')
                            Importancia = pd.DataFrame({'Variable': list(datosArbolesDecision[datosADeciR]),
                                    'Importancia': PronosticoAD.feature_importances_}).sort_values('Importancia', ascending=False)
                            st.table(Importancia)


                            import graphviz
                            from sklearn.tree import export_graphviz
                            # Se crea un objeto para visualizar el árbol
                            # Se incluyen los nombres de las variables para imprimirlos en el árbol
                            st.subheader('Árbol de decisión')
                            from sklearn.tree import plot_tree
                            if st.checkbox('Visualizar árbol de decisión (puede tardar un poco)'):
                                with st.spinner('Generando árbol de decisión...'):
                                    plt.figure(figsize=(16,16))  
                                    plot_tree(PronosticoAD, feature_names = list(datosArbolesDecision[datosADeciR]))
                                    st.pyplot()

                            from sklearn.tree import export_text
                            if st.checkbox('Visualizar árbol en formato de texto: '):
                                Reporte = export_text(PronosticoAD, feature_names = list(datosArbolesDecision[datosADeciR]))
                                st.text(Reporte)

                            Elementos = export_graphviz(PronosticoAD, feature_names = list(datosArbolesDecision[datosADeciR]))  
                            Arbol = graphviz.Source(Elementos)
                            Arbol.format = 'svg'
                            Arbol.render('ArbolDecisionR')
                            with open("ArbolDecisionR.svg", "rb") as file:
                                btn = st.download_button(
                                label="Haz click aquí para descargar el árbol de decisión generado (extensión SVG)",
                                data=file,
                                file_name="ArbolDecisionR.svg",
                                mime="image/svg"
                                )

                            st.markdown("### **El árbol generado se puede leer en el siguiente orden:** ")
                            st.markdown("""
                            1. La decisión que se toma para dividir el nodo.
                            2. El tipo de criterio que se utilizó para dividir cada nodo.
                            3. Cuántos valores tiene ese nodo.
                            4. Valores promedio.
                            5. Por último, el valor pronosticado en ese nodo. """)

                            st.subheader('Pronóstico basado en el modelo establecido')
                            with st.expander("Da click aquí para pronosticar los datos que gustes"):
                                st.subheader('Predicción de casos')
                                sujetoN = st.text_input("Ingrese el nombre o ID del paciente que se desea pronosticar: ")
                                dato = []
                                for p in range(len(datosADeciR)):
                                    dato.append(st.number_input(datosADeciR[p][:], step=0.1))
                                
                                if st.checkbox("Dar pronóstico: "):
                                    resultado = PronosticoAD.predict([dato])[0]
                                    st.info("Con un algoritmo que tiene una exactitud promedio del: "+str(r2_score(Y_test, Y_Pronostico).round(6)*100)+"%, el pronóstico de la variable '"+str(variablePronostico)+"' fue de "+str(resultado)+" para el paciente: "+str(sujetoN)+".")
                        except:
                            st.warning("Por favor, selecciona parámetros válidos para el árbol de decisión")

                    elif X.size == 0:
                        st.warning("No se ha seleccionado ninguna variable")

            if opcionArbol1O2 == "Árbol de decisión (Clasificación)":
                opcionVisualizacionArbolD = st.select_slider('Selecciona una opción', options=["Evaluación Visual", "Matriz de correlaciones","Aplicación del algoritmo"], value="Evaluación Visual")

                if opcionVisualizacionArbolD == "Evaluación Visual":
                    st.dataframe(datosArbolesDecision)
                    st.header("Visualización de los datos")
                    variablePronostico = st.selectbox("Variable a clasificar", datosDelPronostico,index=1)
                    st.write(datosArbolesDecision.groupby(variablePronostico).size())

                    # Seleccionar los datos que se quieren visualizar
                    try:
                        datos = st.multiselect("Datos", datosDelPronostico, default=[datosDelPronostico[2], datosDelPronostico[3]])
                        dato1=datos[0][:]
                        dato2=datos[1][:]

                        if st.checkbox("Gráfico de dispersión: "):
                            with st.spinner("Cargando gráfico de dispersión..."):
                                sns.scatterplot(x=dato1, y=dato2, data=datosArbolesDecision, hue=variablePronostico)
                                plt.title('Gráfico de dispersión')
                                plt.xlabel(dato1)
                                plt.ylabel(dato2)
                                st.pyplot() 

                    except:
                        st.warning("Selecciona solo dos datos")

                    if st.checkbox("Matriz de correlaciones con el propósito de seleccionar variables significativas (puede tardar un poco): "):
                        with st.spinner("Cargando matriz de correlaciones..."):
                            sns.pairplot(datosArbolesDecision, hue=variablePronostico)
                            st.pyplot()
                    
                if opcionVisualizacionArbolD == "Matriz de correlaciones":
                    MatrizCorr = datosArbolesDecision.corr(method='pearson')
                    st.header("Matriz de correlaciones: ")
                    st.dataframe(MatrizCorr)

                    # SELECCIONAR VARIABLES PARA PRONOSTICAR
                    #try:
                        #st.subheader("Correlación de variables: ")
                        #variableCorrelacion = st.selectbox("", MatrizCorr.columns) 
                        #st.table(MatrizCorr[variableCorrelacion].sort_values(ascending=False)[:10])  #Top 10 valores
                    #except:
                        #st.warning("Selecciona una variable con datos válidos...")

                    # Mapa de calor de la relación que existe entre variables
                    st.header("Mapa de calor de la correlación entre variables: ")
                    plt.figure(figsize=(14,7))
                    MatrizInf = np.triu(MatrizCorr)
                    sns.heatmap(MatrizCorr, cmap='RdBu_r', annot=True, mask=MatrizInf)
                    plt.title('Mapa de calor de la correlación que existe entre variables')
                    st.pyplot()

                if opcionVisualizacionArbolD == "Aplicación del algoritmo":
                    st.header('Definición de variables predictoras (X) y variable clase (Y)')
                    MatrizCorr = datosArbolesDecision.corr(method='pearson')
                    st.subheader("Recordando la matriz de correlaciones: ")
                    st.dataframe(MatrizCorr)

                    # SELECCIONAR VARIABLES A CLASIFICAR
                    st.subheader('Selección de la variable Clase')
                    st.markdown('La variable clase debe contener valores **DISCRETOS**')
                    variablePronostico = st.selectbox("Variable a clasificar", datosArbolesDecision.columns,index=1)

                    # Comprobando que la variable clase sea binaria
                    if datosArbolesDecision[variablePronostico].nunique() == 2:
                        col1, col2, col3 = st.columns(3)
                        # Comprobando el tipo de dato de la variable clase
                        if type(datosArbolesDecision[variablePronostico].value_counts().index[1]) and type(datosArbolesDecision[variablePronostico].value_counts().index[0]) != np.int64:
                            with st.expander("Click para ver el dataframe original: "):
                                st.subheader("Dataframe original: ")
                                st.dataframe(datosArbolesDecision)
                            
                            col1.info("Selecciona las etiquetas que gustes...")
                            col2.info("Etiqueta: "+str(datosArbolesDecision[variablePronostico].value_counts().index[0]))
                            col3.info("Etiqueta: "+str(datosArbolesDecision[variablePronostico].value_counts().index[1]))
                            binario1 = col2.text_input("", datosArbolesDecision[variablePronostico].value_counts().index[0])
                            binario2 = col3.text_input("", datosArbolesDecision[variablePronostico].value_counts().index[1])

                            col2.warning("La etiqueta '"+str(datosArbolesDecision[variablePronostico].value_counts().index[0])+"', cambió por la etiqueta: "+binario1)
                            col3.warning("La etiqueta '"+str(datosArbolesDecision[variablePronostico].value_counts().index[1])+"', cambió por la etiqueta: "+binario2)

                            with st.expander("Click para ver el nuevo dataframe: "):
                                st.subheader("Dataframe corregido: ")
                                datosArbolesDecision = datosArbolesDecision.replace({str(datosArbolesDecision[variablePronostico].value_counts().index[1]): binario2, str(datosArbolesDecision[variablePronostico].value_counts().index[0]): binario1})
                                st.dataframe(datosArbolesDecision)
                                Y = np.array(datosArbolesDecision[variablePronostico])
                            
                        # Variables predictoras 
                        st.subheader('Variables Predictoras')
                        # Seleccionar los datos que se quieren visualizar
                        datos = st.multiselect("Selecciona las variables predictoras", datosArbolesDecision.columns.drop(variablePronostico))
                        X = np.array(datosArbolesDecision[datos]) 
                        if X.size > 0:
                            with st.expander("Da click aquí para visualizar el dataframe de las variables predictoras que seleccionaste:"):
                                st.dataframe(X)
                        
                            try:
                                from sklearn.tree import DecisionTreeClassifier
                                from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
                                from sklearn import model_selection
                                # Aplicación del algoritmo: Regresión Logística
                                st.header('Criterio de división')
                                testSize = st.slider('Selecciona el tamaño del test', min_value=0.2, max_value=0.3, value=0.2, step=0.01)
                                X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=testSize, random_state=0, shuffle=True)
                                # Datos de entrenamiento: 70, 75 u 80% de los datos
                                # Datos de prueba: 20, 25 o 30% de los datos

                                # DIVISIÓN DE LOS DATOS EN ENTRENAMIENTO Y PRUEBA
                                #st.dataframe(X_train)
                                #st.dataframe(Y_train)
                                # SE ENTRENA EL MODELO A TRAVÉS DE UN ÁRBOL DE DECISIÓN (REGRESIÓN)
                                # EXPLICACIÓN 
                                st.header('Parámetros del árbol de decisión: ')
                                st.markdown("""
                                * **max_depth**. Indica la máxima profundidad a la cual puede llegar el árbol. Esto ayuda a combatir el overfitting, pero también puede provocar underfitting.
                                * **min_samples_leaf**. Indica la cantidad mínima de datos que debe tener un nodo hoja.
                                * **min_samples_split**. Indica la cantidad mínima de datos para que un nodo de decisión se pueda dividir. Si la cantidad no es suficiente este nodo se convierte en un nodo hoja.
                                * **criterion**. Indica la función que se utilizará para dividir los datos. Puede ser (ganancia de información) gini y entropy (Clasificación). Cuando el árbol es de regresión se usan funciones como el error cuadrado medio (MSE). """)

                                try: 
                                    st.write("Selecciona los valores que requieras para entrenar el modelo: ")
                                    choiceProfuncidad = st.select_slider('Máxima profundidad del árbol (max_depth)', options=["None","Valores numéricos"], value="None")
                                    column1, column2, column3 = st.columns(3)
                                    if choiceProfuncidad == "None":
                                        Max_depth = None
                                    elif choiceProfuncidad == "Valores numéricos":
                                        Max_depth = column1.number_input('Máxima profundidad del árbol (max_depth)', min_value=1, value=8)

                                    Min_samples_split = column2.number_input('min_samples_split', min_value=1, value=2)
                                    Min_samples_leaf = column3.number_input('min_samples_leaf', min_value=1, value=1)
                                    Criterio = st.selectbox('criterion', options=("gini", "entropy"), index=0)
                                    
                                    # Se entrena el modelo a partir de los datos de entrada
                                    ClasificacionAD = DecisionTreeClassifier(criterion=Criterio, max_depth=Max_depth, min_samples_split=Min_samples_split, min_samples_leaf=Min_samples_leaf,random_state=0)
                                    ClasificacionAD.fit(X_train, Y_train)

                                    #Se etiquetan las clasificaciones
                                    Y_Clasificacion = ClasificacionAD.predict(X_validation)
                                    st.markdown('Se etiquetan las clasificaciones (Real vs Clasificado)')
                                    Valores = pd.DataFrame(Y_validation, Y_Clasificacion)
                                    st.dataframe(Valores)


                                    # Matriz de clasificación
                                    st.subheader('Matriz de clasificación')
                                    Y_Clasificacion = ClasificacionAD.predict(X_validation)
                                    Matriz_Clasificacion = pd.crosstab(Y_validation.ravel(), Y_Clasificacion, rownames=['Real'], colnames=['Clasificación'])
                                    #st.table(Matriz_Clasificacion)
                                    
                                    col1, col2 = st.columns(2)
                                    col1.info('Verdaderos Positivos (VP): '+str(Matriz_Clasificacion.iloc[1,1]))
                                    col2.info('Falsos Negativos (FN): '+str(Matriz_Clasificacion.iloc[1,0]))
                                    col2.info('Verdaderos Negativos (VN): '+str(Matriz_Clasificacion.iloc[0,0]))
                                    col1.info('Falsos Positivos (FP): '+str(Matriz_Clasificacion.iloc[0,1]))

                                    # Reporte de clasificación
                                    st.subheader('Reporte de clasificación')
                                    with st.expander("Da click aquí para ver el reporte de clasificación"):
                                        #st.write(classification_report(Y_validation, Y_Clasificacion))
                                        st.success("Criterio: "+str(ClasificacionAD.criterion))
                                        importancia = ClasificacionAD.feature_importances_.tolist()
                                        
                                        st.success("Importancia de las variables: "+str(importancia))
                                        st.success("Exactitud promedio de la validación: "+ str(ClasificacionAD.score(X_validation, Y_validation).round(6)*100)+" %")
                                        precision = float(classification_report(Y_validation, Y_Clasificacion).split()[10])*100
                                        st.success("Precisión: "+ str(precision)+ "%")
                                        st.error("Tasa de error: "+str((1-ClasificacionAD.score(X_validation, Y_validation))*100)+"%")
                                        sensibilidad = float(classification_report(Y_validation, Y_Clasificacion).split()[11])*100
                                        st.success("Sensibilidad: "+ str(sensibilidad)+ "%")
                                        especificidad = float(classification_report(Y_validation, Y_Clasificacion).split()[6])*100
                                        st.success("Especificidad: "+ str(especificidad)+"%")
                                    

                                    st.subheader('Importancia de las variables')
                                    Importancia = pd.DataFrame({'Variable': list(datosArbolesDecision[datos]),
                                            'Importancia': ClasificacionAD.feature_importances_}).sort_values('Importancia', ascending=False)
                                    st.table(Importancia)


                                    import graphviz
                                    from sklearn.tree import export_graphviz
                                    # Se crea un objeto para visualizar el árbol
                                    # Se incluyen los nombres de las variables para imprimirlos en el árbol
                                    st.subheader('Árbol de decisión (Clasificación)')

                                    from sklearn.tree import plot_tree
                                    if st.checkbox('Visualizar árbol de decisión (puede tardar un poco)'):
                                        with st.spinner('Generando árbol de decisión...'):
                                            plt.figure(figsize=(16,16))  
                                            plot_tree(ClasificacionAD, feature_names = list(datosArbolesDecision[datos]),class_names=Y_Clasificacion)
                                            st.pyplot()

                                    from sklearn.tree import export_text
                                    if st.checkbox('Visualizar árbol en formato de texto: '):
                                        Reporte = export_text(ClasificacionAD, feature_names = list(datosArbolesDecision[datos]))
                                        st.text(Reporte)

                                    Elementos = export_graphviz(ClasificacionAD, feature_names = list(datosArbolesDecision[datos]), class_names=Y_Clasificacion)
                                    Arbol = graphviz.Source(Elementos)
                                    Arbol.format = 'svg'
                                    Arbol.render('ArbolDecisionC')
                                    with open("ArbolDecisionC.svg", "rb") as f:
                                        btn = st.download_button(
                                        label="Haz click para descargar el árbol de decisión generado (extensión SVG)",
                                        data=f,
                                        file_name="ArbolDecisionC.svg",
                                        mime="image/svg"
                                        )

                                    st.markdown("### **El árbol generado se puede leer en el siguiente orden:** ")
                                    st.markdown("""
                                    1. La decisión que se toma para dividir el nodo.
                                    2. El tipo de criterio que se usó para dividir cada nodo.
                                    3. Cuantos valores tiene ese nodo.
                                    4. Valores promedio.
                                    5. Por último, el valor clasificado en ese nodo. """)


                                    st.subheader('Clasificación de datos basado en el modelo establecido')
                                    with st.expander("Da click aquí para clasificar los datos que gustes"):
                                        st.subheader('Clasificación de casos')
                                        sujetoN = st.text_input("Ingrese el nombre o ID del sujeto que desea clasificar: ")

                                        dato = []
                                        for p in range(len(datos)):
                                            dato.append(st.number_input(datos[p][:], step=0.1))
                                        
                                        if st.checkbox("Dar clasificación: "):
                                            if ClasificacionAD.predict([dato])[0] == binario2:
                                                st.error("Con un algoritmo que tiene una exactitud del: "+str(round(ClasificacionAD.score(X_validation, Y_validation)*100,2))+"%, la clasificación para el paciente: "+str(sujetoN)+" fue de 0 (CERO), es decir, el diagnóstico fue "+str(binario2).upper())
                                            elif ClasificacionAD.predict([dato])[0] == binario1:
                                                st.success("Con un algoritmo que tiene una exactitud del: "+str(round(ClasificacionAD.score(X_validation, Y_validation)*100,2))+ "%, la clasificación para el paciente: "+str(sujetoN)+" fue de 1 (UNO), es decir, el diagnóstico fue "+str(binario1).upper())
                                            else:
                                                st.warning("El resultado no pudo ser determinado, intenta hacer una buena selección de variables")

                                except:
                                    st.warning("No se pudo entrenar el modelo porque no se seleccionaron parámetros válidos para el árbol de decisión...")
                                
                            except:
                                st.warning("No se pudo realizar la clasificación porque no se ha hecho una correcta selección de variables")

                        elif X.size == 0:
                            st.warning("No se ha seleccionado ninguna variable")

                    elif datosArbolesDecision[variablePronostico].nunique() != 2:
                        st.warning("Por favor, selecciona una variable Clase (a clasificar) que contenga valores binarios...")

if __name__ == "__main__":
    main()

# Para ejeuctarlo en la terminal:
# activate IA
# streamlit run app.py
