# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 20:01:56 2023

    May 13th Version. 1.13.1
    
    TOP 10000 songs streamed in Spotify
    https://www.kaggle.com/datasets/rakkesharv/spotify-top-10000-streamed-songs

@author: DavidXanxez
"""

# import numpy as np
# from sklearn import metrics
# from sklearn.cluster import KMeans
# import pandas as pd
# import nltk
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
# from nltk.tokenize import word_tokenize
# from nltk.stem import WordNetLemmatizer
# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# #Loading MUSIC INFO dataframe/database link: https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019?resource=download
# music_db = pd.read_csv(r"C:\Users\34653\Documents\UPF\TFG\Code\data\tcc_ceds_music.csv")
# music_db.dropna()

# #-------------------------------------------------------------------------------------------------------------

# # Crear una instancia de WordNetLemmatizer
# lemmatizer = WordNetLemmatizer()
# def wordNet_lemmatizer(lyrics):
#     tokens = word_tokenize(lyrics)
#     lemmas = [lemmatizer.lemmatize(token) for token in tokens]
#     lemmatized_lyrics = " ".join(lemmas)
#     return lemmatized_lyrics

# def manual_lemmatizer():
#     # Limpiar Lyrics de las canciones y aplicar lemmatizacion para obtener la raiz de las palabras.
#     music_db['lyrics'] = music_db['lyrics'].apply(lambda x: x.lower())          # convertir todo el texto a minúsculas
#     music_db['lyrics'] = music_db['lyrics'].str.replace('[^\w\s]','')           # eliminar signos de puntuación
#     music_db['lyrics'] = music_db['lyrics'].str.replace('\d+', '')              # eliminar números
#     music_db['lyrics_lemmatized'] = music_db['lyrics'].apply(wordNet_lemmatizer(music_db("lyrics")))  #lemmatizacion usando la funcion lemmatize_lyrics


# #-------------------------------------------------------------------------------------------------------------


# def sentimentAnalysis(lyrics):
#     # # extraer los sentimientos usando la librería Vader
    
#     sia = SentimentIntensityAnalyzer()
    
#     music_db['sentiments'] = music_db['lyrics'].apply(lambda x: sia.polarity_scores(x))
#     music_db['positive_sentiment'] = music_db['sentiments'].apply(lambda x: x['pos'])
#     music_db['negative_sentiment'] = music_db['sentiments'].apply(lambda x: x['neg'])
#     music_db['neutral_sentiment'] = music_db['sentiments'].apply(lambda x: x['neu'])
    
#     # extraer las pulsaciones por minuto
#     #music_db['bpm'] = music_db['bpm'].fillna(data['bpm'].median())
    
    
#     #--------------------------------------------------------------------------------------------------------------
    
#     # seleccionar las características que se utilizarán para el clustering
#     sentiment_classification = music_db[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']]
    
#     # entrenar el modelo de clustering
#     kmeans = KMeans(n_clusters=10)
#     kmeans.fit(sentiment_classification)
#     music_db['cluster'] = kmeans.predict(sentiment_classification)

#--------------------------------------------------------------------------------------------------------------


# # seleccionar las canciones favoritas del usuario
# favorite_songs = music_db[music_db['track_name'].isin(['Cancion1', 'Cancion2', 'Cancion3'])]

# # obtener el clúster de las canciones favoritas
# favorite_cluster = favorite_songs['cluster'].unique()

# # seleccionar las canciones del mismo clúster y ordenarlas según su popularidad
# recommended_songs = music_db[music_db['cluster']] == favorite_cluster

# topics = music_db.filter(items = ["topic"])
# lyrics = music_db.filter(items = ["lyrics"])


# diff_topics = music_db['topic'].drop_duplicates()
# #print(diff_topics)




# for i in lyrics:
#     print(lyrics[i])
#     #analyze each row with sentiment analysis
#     #and then check whether a song can be related with the topic of itself. Otherwise study the case.








import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors


import tensorflow as tf

import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import random

# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Carga de datos
def load_data():
    #Loading MUSIC INFO dataframe/database link: https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019?resource=download
    music_db = pd.read_csv(r"C:\Users\34653\Documents\UPF\TFG\Code\data\tcc_ceds_music.csv")
    music_db.dropna(inplace=True)
    
    music_db.drop(columns=['Unnamed: 0'], inplace=True)
    

    #millSong_db = pd.read_csv(r"C:\Users\34653\Documents\UPF\TFG\Code\data\spotify_millsongdata.csv")
    #millSong_db.dropna(inplace=True)
    
    return music_db#, millSong_db


def encoding_and_normalization(dataframe):
    
    #ENCODING
    le = LabelEncoder()   
    music_db['genre_code'] = le.fit_transform(music_db['genre'])
    music_db['release_date_code'] = le.fit_transform(music_db['release_date'])
    music_db['topic_code'] = le.fit_transform(music_db['topic'])

    #NORMALIZATION
    scaler = MinMaxScaler()
    columns_to_normalize = ['genre_code', 'release_date_code', 'len', 'topic_code']
    music_db[columns_to_normalize] = scaler.fit_transform(music_db[columns_to_normalize])
    
    

# Limpieza y lematización de las letras de las canciones
def clean_lyrics(dataframe):
    # Crear una instancia de WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    def lemmatize_lyrics(lyrics):
        tokens = word_tokenize(lyrics)
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        lemmatized_lyrics = " ".join(lemmas)
        return lemmatized_lyrics

    # Limpiar Lyrics de las canciones y aplicar lemmatizacion para obtener la raiz de las palabras.
    dataframe['lyrics'] = dataframe['lyrics'].apply(lambda x: x.lower())          # convertir todo el texto a minúsculas
    dataframe['lyrics'] = dataframe['lyrics'].str.replace('[^\w\s]','')           # eliminar signos de puntuación
    dataframe['lyrics'] = dataframe['lyrics'].str.replace('\d+', '')              # eliminar números
    dataframe['lyrics_lemmatized'] = dataframe['lyrics'].apply(lemmatize_lyrics)  #lemmatizacion usando la funcion lemmatize_lyrics
    return dataframe



def analyze_liked_songs(liked_songs, dataframe, sentiment_plot=False, year_genre_plot= False):
    
    songs_df = dataframe[dataframe['track_name'].isin(liked_songs)].drop_duplicates(subset=['track_name'])
    #year = get_mean_year(songs_df)

    #genre_counts = songs_df.groupby('genre').size()
    #predominant_genre = genre_counts.idxmax()

    songs_df[['positive_sentiment', 'negative_sentiment']] = songs_df[['positive_sentiment', 'negative_sentiment']].apply(pd.to_numeric)
    songs_df[['release_date']] = songs_df[['release_date']].apply(pd.to_numeric)
    
    
    if sentiment_plot:
       plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'sentiment_cluster']],
                     dataframe['sentiment_cluster'], 
                     n_clusters_sentiment, 
                     recommended_songs=None, 
                     liked_songs=songs_df,
                     sentiment=True,
                     title="Sentiment Liked Songs")
    
    if year_genre_plot:
        plot_clusters(dataframe[['genre', 'release_date', 'cluster_genre_year']], 
                      dataframe['cluster_genre_year'], 
                      n_clusters_year_genre, 
                      recommended_songs=None, 
                      liked_songs=songs_df,
                      genre_year=True,
                      title="Year/Genre Liked Songs")

    return songs_df



def create_liked_songs_df(liked_songs, music_db):
    
    # Buscar las canciones en el music_db y obtener sus datos
    song_data = []
    
    for title in liked_songs:
        song = music_db.query("track_name == @title")
        if len(song) > 0:
            song_data.append(song.iloc[0])
        else:
            print(f"No se ha encontrado la canción: {song}")
            
    # Crear el DataFrame liked_songs_df
    if len(song_data) > 0:
        liked_songs_df = pd.DataFrame(song_data)
        return liked_songs_df
    else:
        return None


# Análisis de sentimientos
def sentiment_analysis(dataframe):
    # extraer los sentimientos usando la librería Vader
    sia = SentimentIntensityAnalyzer()
    dataframe['sentiments'] = dataframe['lyrics'].apply(lambda x: sia.polarity_scores(x))
    dataframe['positive_sentiment'] = dataframe['sentiments'].apply(lambda x: x['pos'])
    dataframe['negative_sentiment'] = dataframe['sentiments'].apply(lambda x: x['neg'])
    dataframe['neutral_sentiment'] = dataframe['sentiments'].apply(lambda x: x['neu'])
    return dataframe


def genre_analysis(dataframe, songs):
    
    unique_genres = dataframe['genre_code'].unique()    #valores de genero validos
    genre_median = songs['genre_code'].median()
    
    closest_genre = min(unique_genres, key=lambda x: abs(x - genre_median))
    
    return closest_genre

def year_analysis(dataframe, songs):
    
    unique_years = dataframe['release_date_code'].unique()    #valores de genero validos
    songs = dataframe.loc[songs.index]
    year_mean = songs['release_date_code'].mean()
    
    closest_year = min(unique_years, key=lambda x: abs(x - year_mean))
    
    return closest_year

def topic_analysis(dataframe, songs):
    
    unique_topic = dataframe['topic_code'].unique()    #valores de genero validos
    songs = dataframe.loc[songs.index]
    topic_median = songs['topic_code'].median()
    
    closest_topic = min(unique_topic, key=lambda x: abs(x - topic_median))
    
    return closest_topic


########################################### CLUSTERING ###################################################

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





############################### PLOT CLUSTERING ######################################

def plot_clusters(dataframe, cluster_col, n_clusters, recommended_songs=None, liked_songs = None, sentiment=False, genre_year=False, title=None):
    colors = plt.cm.get_cmap('tab10', n_clusters)
    
    if sentiment:
        if 'sentiment_cluster' in dataframe:
            plt.figure(figsize=(10,8))
            for i in range(len(dataframe['sentiment_cluster'].unique())):            
                cluster_df = dataframe[dataframe['sentiment_cluster'] == i]
                plt.scatter(cluster_df['positive_sentiment'],
                            cluster_df['negative_sentiment'], 
                            color=colors(i), label=f'Cluster {i}')
            
            if recommended_songs is not None:
                plt.scatter(recommended_songs['positive_sentiment'], 
                            recommended_songs['negative_sentiment'], 
                            color='black', marker='x',  s= 100, label='Recommended songs')
            
            if liked_songs is not None:
                liked_songs[['positive_sentiment', 'negative_sentiment']] = liked_songs[['positive_sentiment', 'negative_sentiment']].apply(pd.to_numeric)
                liked_songs_subset = dataframe[dataframe.index.isin(liked_songs.index)]
                plt.scatter(liked_songs_subset['positive_sentiment'],
                            liked_songs_subset['negative_sentiment'], 
                            color='black', marker='o',  s= 100, label='Liked songs')
    
            plt.title(title)
            plt.xlabel('positive_sentiment')
            plt.ylabel('negative_sentiment')
            plt.legend()
            plt.show()
        else:
            print("La columna 'cluster' no existe en el dataframe.")
            
    if genre_year:
        if 'cluster_genre_year' in dataframe:
            plt.figure(figsize=(10,8))
            for i in range(len(dataframe['cluster_genre_year'].unique())):            
                cluster_df = dataframe[dataframe['cluster_genre_year'] == i]
                plt.scatter(cluster_df['genre'],
                            cluster_df['release_date'], 
                            color=colors(i), label=f'Cluster {i}')
            
            if recommended_songs is not None:
                plt.scatter(recommended_songs['genre'], 
                            recommended_songs['release_date'], 
                            color='black', marker='x', s= 100, label='Recommended songs')
            
            if liked_songs is not None:
                liked_songs[['release_date']] = liked_songs[['release_date']].apply(pd.to_numeric)
                liked_songs_subset = dataframe[dataframe.index.isin(liked_songs.index)]
                plt.scatter(liked_songs_subset['genre'],
                            liked_songs_subset['release_date'], 
                            color='black', marker='o', s= 100, label='Liked songs')
    
            plt.title(title)
            plt.xlabel('genre')
            plt.ylabel('release_date')
            plt.legend()
            plt.show()
        else:
            print("La columna 'cluster_genre_year' no existe en el dataframe.")
        
        
        


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
           plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'sentiment_cluster']], 
                         dataframe['sentiment_cluster'], 
                         n_clusters_sentiment, 
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
           plot_clusters(dataframe[['genre', 'release_date', 'cluster_genre_year']],
                         dataframe['cluster_genre_year'], 
                         n_clusters_year_genre, 
                         year_genre_recommended_songs,
                         genre_year=True,
                         title="Genre/Year Recommended Songs")   
           
           return year_genre_recommended_songs



def generalRecommendations(liked_songs_df, dataframe, plot_recommendations=False):
    
    # quitar aquellas columnas que no puedan producir un calculo en NumPy, y pasamos los demas valores a numpy
    liked_songs_data = liked_songs_df.drop(['artist_name', 'track_name', 'lyrics', 'genre', 'release_date', 'topic', 'lyrics_lemmatized', 'sentiments' ], axis=1).to_numpy()

    # calcular la media de las columnas de las liked_songs
    column_means = np.mean(liked_songs_data, axis=0) #axis=0 de arriba a abajo

    #calcular el numero de veces que cada columna aparece en las canciones que le gustan
    column_counter = np.sum(liked_songs_data <= 0, axis=0)
    column_weights = np.zeros(column_means.shape)
    
    # calcular los pesos de las columnas
    for i in range(column_weights.shape[0]):
        if column_counter[i] > 0:
            column_weights[i] = column_counter[i] / column_means[i]

    # normalizar los pesos de las columnas
    column_weights = column_weights / np.sum(column_weights)

    # calcular el cosine similarity con los pesos de las columnas ajustados
    music_data = dataframe.drop(['artist_name', 'track_name', 'lyrics', 'genre', 'release_date' ,'topic', 'lyrics_lemmatized', 'sentiments'], axis=1).to_numpy()
    
    #print(dataframe.columns)
    
    weighted_music_data = music_data * column_weights.reshape(1, -1)
    weighted_liked_songs_data = liked_songs_data * column_weights.reshape(1, -1)
    similarity_scores = cosine_similarity(weighted_music_data, weighted_liked_songs_data)

    # obtener los índices de las canciones más similares
    index = np.argsort(similarity_scores, axis=0)[::-1][:10]
    indexes_recommendation= np.unique(index)

    # crear un dataframe con las canciones recomendadas
    recommended_songs = dataframe.iloc[indexes_recommendation]
    
    # if plot_recommendations NO TIENE SENTIDO HACER UN PLOT PORQUE NO HAY CLUSTERIZACION
    #    plot_clusters(dataframe[column_weights[:2]],
    #                  dataframe['cluster_genre_year'], 
    #                  n_clusters, 
    #                  recommended_songs,
    #                  title="General Recommended Songs")   
    
    return recommended_songs


def kNNRecommendation(dataframe, analyzed_songs, k, num_recommendations = 10):
    
    genre_mean = genre_analysis(dataframe, analyzed_songs)
    release_date_mean = year_analysis(dataframe, analyzed_songs)
    topic_median = topic_analysis(dataframe, analyzed_songs)
    
    
    songs_features = analyzed_songs[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']].to_numpy()
    dataframe_features = dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment']].to_numpy()
    
    #knn analisis
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(songs_features)

    # Encontrar los k vecinos más cercanos
    _, indices = knn.kneighbors(dataframe_features)
    
    # Obtener las canciones recomendadas por los vecinos más cercanos
    recommended_songs_indices = np.unique(indices.flatten())  # Indices únicos de las canciones recomendadas
    recommended_songs = dataframe.iloc[recommended_songs_indices]

    # Filtrar canciones por género y año
    recommended_songs = dataframe.loc[(dataframe['genre_code'] == genre_mean) & 
                                      (dataframe['release_date_code'] == release_date_mean) & 
                                      (dataframe['topic_code'] == topic_median)]

    if len(recommended_songs) > num_recommendations:
        recommended_songs = random.sample(list(recommended_songs.index), num_recommendations)
        recommended_songs = dataframe.loc[recommended_songs]


    plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'sentiment_cluster']], 
                  dataframe['sentiment_cluster'], 
                  n_clusters_sentiment, 
                  recommended_songs, 
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
    song_analysis = analyze_liked_songs(merged_songs, dataframe)
    merged_recommendation = recommend_songs(dataframe, song_analysis)
    
 
    plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'sentiment_cluster']],
                  dataframe['sentiment_cluster'], 
                  n_clusters_sentiment, 
                  recommended_songs=merged_recommendation,
                  sentiment=True,
                  title= "Merged Sentiment Recommended Songs")
    
    plot_clusters(dataframe[['genre', 'release_date', 'cluster_genre_year']], 
                dataframe['cluster_genre_year'], 
                n_clusters_year_genre, 
                recommended_songs=merged_recommendation,
                genre_year=True,
                title= "Merged Year/Genre Recommended Songs")
    
    
    return merged_recommendation
    
   
def printSongs(recommended_songs, liked_songs_trigger=False):
    recommended_songs = recommended_songs['track_name'] + ' - ' +  recommended_songs['artist_name']
    liked_songs = liked_songs_df['track_name'] + ' - ' +  liked_songs_df['artist_name']
    
    if liked_songs_trigger:
        print("\n")
        print("Your liked songs are: ")
        for i, song in enumerate(liked_songs):
            print(f"{i+1}. {song}")
    
    print("\n")
    print("The general recommended songs are: " )
    for i, song in enumerate(recommended_songs):
        print(f"{i+1}. {song}")





if __name__ == '__main__':
    n_clusters = 10
    n_clusters_sentiment= 10
    n_clusters_year_genre = 10 
    k = 3
    liked_songs = ['young, wild & free (feat. bruno mars)', 'riptide', "delight for old chicken"]

    # Ejecutar funciones
    music_db = load_data()
    #music_db = music_db.sample(n=10000, random_state=1)
    
    music_db = clean_lyrics(music_db)
    music_db = sentiment_analysis(music_db)
    encoding_and_normalization(music_db)
    
    #database sentimentally clustered
    music_db = sentiment_cluster_songs(music_db, n_clusters_sentiment)
    music_db = cluster_songs_by_genre_year(music_db, n_clusters_year_genre)
    
    #database clustered 
    liked_songs_df = create_liked_songs_df(liked_songs, music_db)
    liked_songs_analyzed = analyze_liked_songs(liked_songs, music_db, sentiment_plot=True, year_genre_plot=True)
    
    #merged_recommended_songs = recommend_songs(music_db, liked_songs_df, plot_recommendations=True)
    #emotionally_recommended_songs = recommend_songs(music_db, liked_songs, plot_recommendations=True, sentiment=True)
    #genre_year_recommended_songs = recommend_songs(music_db, liked_songs, plot_recommendations=True, genre_year=True)
    #merged_emotion_genre_year = mergeRecommendations(music_db, emotionally_recommended_songs, genre_year_recommended_songs)
    #general_recommendation_songs = generalRecommendations(liked_songs_df, music_db, plot_recommendations=False)
    knn_recommendation_songs = kNNRecommendation(music_db, liked_songs_df, k)
    
    #printSongs(merged_recommended_songs, liked_songs_trigger=True)
    #printSongs(emotionally_recommended_songs)
    #printSongs(genre_year_recommended_songs)
    #printSongs(merged_emotion_genre_year)
    #printSongs(general_recommendation_songs)
    printSongs(knn_recommendation_songs)
    




