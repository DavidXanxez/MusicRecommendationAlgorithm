
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.neighbors import NearestNeighbors


import tensorflow as tf

import pandas as pd
import random

from collections import Counter

import data
import plot



# Recomendación de canciones
def recommend_songs(dataframe, analyzed_songs, plot_recommendations=False, sentiment=False, genre_year=False):
    
    #analisis de las canciones para recomendar en funcion de ellas
    #year, genre = analyze_liked_songs(liked_songs, dataframe, plot=True)
    
    favorite_songs = dataframe[dataframe['track_name'].isin(analyzed_songs)]
    
    #si no se especifica un filtro en concreto se buscan las mas parecidas
    if (sentiment == False and genre_year == False):     
        # Filtrar las canciones no escuchadas

        unplayed_songs = dataframe[~dataframe.index.isin(analyzed_songs.index.tolist())]

        similarity_matrix = cosine_similarity(unplayed_songs[['genre_code']], analyzed_songs[['genre_code']])

        similar_indices = np.argsort(-similarity_matrix.mean(axis=1))[:10] #seleccionamos las ultimas 10, que seran las que tengan los valores maximos de similitud
        recommended_songs = unplayed_songs.iloc[similar_indices]
           
        return recommended_songs
           
        
    #recommended_songs = dataframe[(dataframe['genre'] == genre) & (dataframe['release_date'] == year)]

    if sentiment:
        favorite_songs_sentiment_cluster = favorite_songs['sentiment_cluster'].unique()
        sentiment_recommended_songs = dataframe[dataframe['sentiment_cluster'].isin(favorite_songs_sentiment_cluster + 1) | dataframe['sentiment_cluster'].isin(favorite_songs_sentiment_cluster - 1)]
        sentiment_recommended_songs = sentiment_recommended_songs[~sentiment_recommended_songs['track_name'].isin(analyzed_songs)].head(10) # Excluir las canciones favoritas

        sentiment_recommended_songs[['positive_sentiment', 'negative_sentiment']] = sentiment_recommended_songs[['positive_sentiment', 'negative_sentiment']].apply(pd.to_numeric)
        if plot_recommendations:
           plot.plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'sentiment_cluster']], 
                         sentiment_recommended_songs, 
                         sentiment=True,
                         title="Sentiment Recommended Songs")
           
           return sentiment_recommended_songs

   
    if genre_year:
        favorite_songs_year_genre_cluster = favorite_songs['cluster_genre_year'].unique()
        #print(favorite_songs_cluster)
        # seleccionar las canciones del mismo genre_year_clúster y seleccionar las que no esten en favorite_songs_name
        
        year_genre_recommended_songs = dataframe[dataframe['cluster_genre_year'].isin(favorite_songs_year_genre_cluster)]
        year_genre_recommended_songs = year_genre_recommended_songs[~year_genre_recommended_songs['track_name'].isin(analyzed_songs)].head(10) # Excluir las canciones favoritas

        year_genre_recommended_songs[['release_date']] = year_genre_recommended_songs[['release_date']].apply(pd.to_numeric)
        if plot_recommendations:
           plot.plot_clusters(dataframe[['genre', 'release_date', 'cluster_genre_year']],
                         year_genre_recommended_songs,
                         genre_year=True,
                         title="Genre/Year Recommended Songs")   
           
           return year_genre_recommended_songs



def cosine_sim_Recommendations(liked_songs_df, dataframe, plot_recommendations=False):
    
    release_date_mean = data.year_analysis(dataframe, liked_songs_df)
    
    liked_genres = set()
    for genres_list in liked_songs_df['genres']:
        if isinstance(genres_list, list):
            liked_genres.update(genres_list)

                
    dataframe_cosine = dataframe[dataframe['genres'].apply(lambda x: any(genre in liked_genres for genre in x)) &
                                 (dataframe['release_date_code'] == float(release_date_mean))]

    
    # Quitar aquellas columnas que no puedan producir un cálculo en NumPy, y convertir los demás valores a numpy
    liked_songs_data = liked_songs_df.drop(['artist_name', 'track_name', 'lyrics', 'genre', 'release_date', 'topic', 'lyrics_lemmatized', 'Total Streams', 'genres'], axis=1)
    liked_songs_data = liked_songs_data.fillna(0).to_numpy()

    # Calcular la media de las columnas de las liked_songs
    column_means = np.mean(liked_songs_data, axis=0)  # axis=0 de arriba a abajo

    # Calcular el número de veces que cada columna aparece en las canciones que le gustan
    column_counter = np.sum(liked_songs_data <= 0, axis=0)
    column_weights = np.zeros(column_means.shape)
    
    # Calcular los pesos de las columnas
    for i in range(column_weights.shape[0]):
        if column_counter[i] > 0:
            column_weights[i] = column_counter[i] / column_means[i]

    # Normalizar los pesos de las columnas
    column_weights = column_weights / np.sum(column_weights)
    

    # Calcular el cosine similarity con los pesos de las columnas ajustados
    music_data = dataframe_cosine.drop(['artist_name', 'track_name', 'lyrics', 'genre', 'release_date', 'topic', 'lyrics_lemmatized', 'Total Streams', 'genres'], axis=1)
    #music_data = music_data.fillna(0).to_numpy()
    
    mask_liked = ~np.all(liked_songs_data == 0, axis=1)
    mask_music = ~np.all(music_data == 0, axis=1)
    liked_songs_data = liked_songs_data[mask_liked]
    music_data = music_data[mask_music]
    
    weighted_music_data = music_data * column_weights.reshape(1, -1)
    weighted_liked_songs_data = liked_songs_data * column_weights.reshape(1, -1)
    
    #drop NaN
    weighted_music_data = np.nan_to_num(weighted_music_data)
    weighted_liked_songs_data = np.nan_to_num(weighted_liked_songs_data)
    
    #print(weighted_liked_songs_data)

    similarity_scores = cosine_similarity(weighted_music_data, weighted_liked_songs_data)

    # Obtener los índices de las canciones más similares
    index = np.argsort(similarity_scores, axis=0)[::-1]
    indexes_recommendation = np.unique(index)
    
    # Crear un dataframe con las canciones recomendadas
    recommended_songs = dataframe_cosine.iloc[indexes_recommendation]

    
    #dataframe_cosine = dataframe[~dataframe['track_name'].isin(liked_songs_df['track_name'])]
    
    
    recommended_songs = recommended_songs[~recommended_songs['track_name'].isin(liked_songs_df['track_name'])]
    #recommended_songs = recommended_songs.sort_values(by='Total Streams', ascending=False).head(10)
    
    plot.plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']], 
                  1,
                  recommended_songs, 
                  liked_songs=liked_songs_df,
                  sentiment=True,
                  title="Cosine Sim Recommended Songs")

    return recommended_songs.head(10)


def kNNRecommendation(dataframe, analyzed_songs, k, num_recommendations = 10):
    
    liked_genres = set()
    for genres_list in analyzed_songs['genres']:
        if isinstance(genres_list, list):
            liked_genres.update(genres_list)

                
    dataframe = dataframe[dataframe['genres'].apply(lambda x: any(genre in liked_genres for genre in x))]

    songs_features = analyzed_songs[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'violence','loudness', 'acousticness', 'energy' ]].to_numpy()
    dataframe_features = dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'violence', 'loudness', 'acousticness', 'energy' ]].to_numpy()
                
    #dataframe = dataframe[dataframe['genres'].apply(lambda x: any( genre in liked_genres for genre in x))] 

    #knn analisis
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(dataframe_features)
    
    # Encontrar los k vecinos más cercanos
    _, indices = knn.kneighbors(songs_features, n_neighbors=k)
    
    # Obtener las canciones recomendadas por los vecinos más cercanos
    recommended_songs_indices = indices.flatten()  # Índices de las canciones recomendadas
    recommended_songs = dataframe.iloc[recommended_songs_indices]
        
    recommended_songs = recommended_songs[~recommended_songs['track_name'].isin(analyzed_songs['track_name'])]    
    recommended_songs = recommended_songs.sort_values(by='Total Streams', ascending=False).sample(10)

    plot.plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']], 
                  1,
                  recommended_songs, 
                  liked_songs= analyzed_songs,
                  sentiment=True,
                  title="Sentiment KNN Recommended Songs")
    
    return recommended_songs


def tensorFlowKmeans(dataframe, analyzed_songs):
    
    # Preprocesamiento de datos
    selected_features = ['track_name', 'release_date', 'lyrics_lemmatized', 'positive_sentiment', 'negative_sentiment', 'valence', 'danceability']
    data = dataframe[selected_features]
    data = data.dropna() # Eliminar filas con valores faltantes
    
    # Transformar datos
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(data['lyrics_lemmatized'])
    lyrics_encoded = tokenizer.texts_to_sequences(data['lyrics_lemmatized'])
    data['lyrics_encoded'] = lyrics_encoded
    
    # Entrenamiento del modelo de clusterización
    kmeans = KMeans(n_clusters=10)
    features = ['positive_sentiment', 'negative_sentiment', 'valence', 'danceability']
    kmeans.fit(data[features])
    
    # Predicción de los clusters para las canciones de entrada
    analyzed_songs['cluster'] = kmeans.predict(analyzed_songs[features])
    
    # Recomendación de canciones del mismo cluster
    recommended_songs = data[data['cluster'] == analyzed_songs.iloc[0]['cluster']]
    recommended_songs = recommended_songs[~recommended_songs['track_name'].isin(analyzed_songs['track_name'])]
    
    return recommended_songs.sample(10)['track_name']




def mergeRecommendations(dataframe, rec_1, rec_2):
    
    merged_songs = pd.concat([rec_1, rec_2]).reset_index(drop=True)
    song_analysis = data.analyze_liked_songs(merged_songs, dataframe)
    merged_recommendation = recommend_songs(dataframe, song_analysis)
    
 
    plot.plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'sentiment_cluster']],
                  recommended_songs=merged_recommendation,
                  sentiment=True,
                  title= "Merged Sentiment Recommended Songs")
    
    plot.plot_clusters(dataframe[['genre', 'release_date', 'cluster_genre_year']],  
                recommended_songs=merged_recommendation,
                genre_year=True,
                title= "Merged Year/Genre Recommended Songs")
    
    
    return merged_recommendation
    
   



