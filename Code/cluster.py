# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:48:43 2023

@author: DavidXanxez
"""


from sklearn.cluster import KMeans
import pandas as pd




# Clustering de canciones
def sentiment_cluster_songs(dataframe, n_clusters, favorite_songs_names=None):
    # seleccionar las características que se utilizarán para el clustering
    X = dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']]

    # entrenar el modelo de clustering
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)
    dataframe['sentiment_cluster'] = kmeans.predict(X)

    if favorite_songs_names is not None:
        # obtener el cluster de las canciones favoritas y añadirlo a la columna 'cluster' del DataFrame
        favorite_songs_clusters = kmeans.predict(dataframe[dataframe['track_name'].isin(favorite_songs_names)][['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']])
        dataframe.loc[dataframe['track_name'].isin(favorite_songs_names), 'sentiment_cluster'] = favorite_songs_clusters

    # plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'sentiment_cluster']],
    #               dataframe['sentiment_cluster'].tolist(), 
    #               n_clusters, 
    #               sentiment=True)
    
    return dataframe


def cluster_songs_by_genre_year(dataframe, n_clusters, favorite_songs_names=None):
    # Seleccionar características para clustering
    X = dataframe[['genre', 'release_date']]
    
    #transformar el genero en un valor numerico para poder hacer la clusterizacion
    X = pd.get_dummies(X, columns=['genre'])

    # Entrenar modelo de clustering
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)

    # Añadir predicciones de cluster al dataframe
    dataframe['cluster_genre_year'] = kmeans.predict(X)
    
    if favorite_songs_names is not None:
        # obtener el cluster de las canciones favoritas y añadirlo a la columna 'cluster' del DataFrame
        favorite_songs_clusters = kmeans.predict(dataframe[dataframe['track_name'].isin(favorite_songs_names)][['genre', 'release_date']])
        dataframe.loc[dataframe['track_name'].isin(favorite_songs_names), 'cluster'] = favorite_songs_clusters

    # Visualizar clusters
    # plot_clusters(dataframe[['genre', 'release_date', 'cluster_genre_year']],
    #               dataframe['cluster_genre_year'].tolist(), 
    #               n_clusters, 
    #               genre_year=True)

    return dataframe



