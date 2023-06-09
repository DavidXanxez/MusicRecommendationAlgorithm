# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:42:30 2023

@author: DavidXanxez
"""


from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import plot
import ast


#Loading data
def load_data():
    #Loading MUSIC INFO dataframe/database link: https://www.kaggle.com/datasets/saurabhshahane/music-dataset-1950-to-2019?resource=download
    music_db = pd.read_csv(r"C:\Users\34653\Documents\UPF\TFG\MusicRecommendationAlgorithm\Code\data\DB_2.0.csv")
    music_db.dropna(inplace=True)
    
    
    music_db.drop(columns=['Unnamed: 0'], inplace=True)
    music_db['track_name'] = music_db['track_name'].str.lower()
    music_db['artist_name'] = music_db['artist_name'].str.lower()
    music_db['track_name'] = music_db['track_name'].str.strip()
    music_db['artist_name'] = music_db['artist_name'].str.strip()
    

    popularity= pd.read_csv(r"C:\Users\34653\Documents\UPF\TFG\MusicRecommendationAlgorithm\Code\data\Spotify_final_dataset.csv")
    popularity.dropna(inplace=True)
    
    popularity['Song Name'] = popularity['Song Name'].fillna('').apply(lambda x: x.lower())
    popularity['Artist Name'] = popularity['Artist Name'].fillna('').apply(lambda x: x.lower())

    popularity['Song Name'] = popularity['Song Name'].str.lower()
    popularity['Song Name'] = popularity['Song Name'].str.strip()
    popularity['Artist Name'] = popularity['Artist Name'].str.lower()
    popularity['Artist Name'] = popularity['Artist Name'].str.strip()
    
    artists = pd.read_csv(r"C:\Users\34653\Documents\UPF\TFG\MusicRecommendationAlgorithm\Code\data\artists.csv")
    artists['name'] = artists['name'].str.lower()
    artists['name'] = artists['name'].str.strip()
    
    artists['genres'] = artists['genres'].apply(ast.literal_eval) 
    
    
    column_names = ['userid', 'timestamp', 'artid', 'artname', 'traid', 'track-name']
    user_db = pd.read_csv(r"C:\Users\34653\Documents\UPF\TFG\MusicRecommendationAlgorithm\Code\data\userid-timestamp-artid-artname-traid-traname.tsv", delimiter='\t', error_bad_lines=False, nrows=8000000, names=column_names)
    user_db['track-name'] = user_db['track-name'].str.lower()
    user_db['track-name'] = user_db['track-name'].str.strip()

    return music_db, popularity, artists, user_db


def merge_datasets(music_db, popularity, artists):
    merged_data = music_db.merge(popularity, left_on=['artist_name', 'track_name'], right_on=['Artist Name', 'Song Name'], how='left')
    merged_data['Total Streams'] = merged_data['Total Streams'].fillna(0)  # Cambiar NaN por 0 en 'Total Streams'

        
    artists_subset = artists[['name', 'genres']]
    merged_data = merged_data.merge(artists_subset, left_on='artist_name', right_on='name', how='left')
    merged_data.drop_duplicates(subset=['artist_name', 'track_name'], inplace=True)
    merged_data['genres'] = merged_data['genres'].fillna('[]')


    merged_data = merged_data[['artist_name', 'track_name', 'release_date', 'genre', 'lyrics', 'len', 'dating', 'violence', 'world/life', 'night/time', 'shake the audience', 'family/gospel', 'romantic', 'communication', 'obscene', 'music', 'movement/places', 'light/visual perceptions', 'family/spiritual', 'like/girls', 'sadness', 'feelings', 'danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy', 'topic', 'lyrics_lemmatized', 'positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'genre_code', 'release_date_code', 'topic_code', 'Total Streams', 'genres']]
    
    merged_data.reset_index(drop=True, inplace=True)
    
    return merged_data



def encoding_and_normalization(dataframe):
    
    #ENCODING
    le = LabelEncoder()   
    dataframe['genre_code'] = le.fit_transform(dataframe['genre'])
    dataframe['release_date_code'] = le.fit_transform(dataframe['release_date'])
    dataframe['topic_code'] = le.fit_transform(dataframe['topic'])
    dataframe['artist_name_code'] = le.fit_transform(dataframe['artist_name'])

    #NORMALIZATION
    scaler = MinMaxScaler()
    columns_to_normalize = ['genre_code', 'release_date_code', 'len', 'topic_code', 'artist_name_code']
    dataframe[columns_to_normalize] = scaler.fit_transform(dataframe[columns_to_normalize])
    
    
    
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
    unique_genres = dataframe['genre_code'].unique()    # Valores de género válidos    
    genre_median = songs['genre_code'].median()

    # Buscar el género más cercano al valor de la mediana
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


def analyze_liked_songs(liked_songs, dataframe, sentiment_plot=False, year_genre_plot= False):
    
    songs_df = dataframe[dataframe['track_name'].isin(liked_songs)].drop_duplicates(subset=['track_name'])
    #year = get_mean_year(songs_df)

    #genre_counts = songs_df.groupby('genre').size()
    #predominant_genre = genre_counts.idxmax()

    songs_df[['positive_sentiment', 'negative_sentiment']] = songs_df[['positive_sentiment', 'negative_sentiment']].apply(pd.to_numeric)
    songs_df[['release_date']] = songs_df[['release_date']].apply(pd.to_numeric)
    
    
    if sentiment_plot:
        plot.plot_clusters(dataframe[['positive_sentiment', 'negative_sentiment']],
                           liked_songs=songs_df, 
                           sentiment=True, genre_year=False, graph_3D=False,
                           title="Sentiment Liked Songs Plot")

    
    if year_genre_plot:
        plot.plot_clusters(dataframe[['genre', 'release_date']], 
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
   
    
def create_song_attributes_df(song_titles, merged_ds):
    # Filtrar las canciones de la base de datos que contengan los títulos de canciones dados
    filtered_songs = merged_ds[merged_ds['track_name'].str.contains('|'.join(song_titles))]
 
    # Crear un nuevo dataframe con las canciones y todos sus atributos
    song_attributes_df = filtered_songs.copy()
    
    return song_attributes_df



def search_correlation(dataframe):
    
    columns =['positive_sentiment', 'negative_sentiment', 'neutral_sentiment',  'violence', 'world/life', 'night/time', 'shake the audience', 'romantic', 'music', 'sadness', 'loudness',  'acousticness', 'energy', 'topic_code', 'genre_code']
    
    # Crear un nuevo dataframe con las columnas seleccionadas
    selected_df = dataframe[columns]
    
    # Calcular la matriz de correlación
    correlation_matrix = selected_df.corr()
    
    # Visualizar la matriz de correlación en un mapa de calor
    
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=0.5)
    
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    
    heatmap.set_xticklabels(columns, fontsize=8)  
    heatmap.set_yticklabels(columns, fontsize=8)
    heatmap.set_title('Heatmap', fontsize=12) 
    plt.show()
    
    # Mostrar el gráfico
    heatmap.figure.savefig('data/heatmap.png', dpi=300, bbox_inches='tight')
    
    
    
def createRecommendationMatrix(dataframe, user_db):
    dataframe['track_name'].fillna('Desconocido', inplace=True)
    
    usuarios_unicos = user_db['userid'].unique()
    canciones_unicas = dataframe['track_name'].str.strip().unique()
    
    print(len(canciones_unicas))
    
    # Crear una matriz vacía de tamaño usuarios x canciones
    matriz_recomendacion = np.zeros((len(usuarios_unicos), len(canciones_unicas)))
    
    # Llenar la matriz con los valores correspondientes
    for _, row in user_db.iterrows():
        usuario = row['userid']
        cancion = row['track-name']
        valor = 1  # Puedes ajustar el valor según tus criterios
        
        #print(usuario, cancion)
        
        # Verificar si la canción está presente en el array de canciones únicas
        if cancion in canciones_unicas:
            # Obtener el índice de la canción en el array de canciones únicas
            cancion_idx = np.where(canciones_unicas == cancion)[0][0]
            
            # Obtener el índice del usuario en la matriz
            usuario_idx = np.where(usuarios_unicos == usuario)[0][0]
            
            # Asignar el valor en la matriz
            matriz_recomendacion[usuario_idx, cancion_idx] = valor
    
    # Ahora tienes la matriz de usuarios y canciones lista para su uso en el sistema de recomendación
    return matriz_recomendacion 

   
    
    
    
    
    
    
    
    