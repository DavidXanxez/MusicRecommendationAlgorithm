# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:38:42 2023

@author: DavidXanxez
"""



# nltk.download('vader_lexicon')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


import data
import algs

import pandas as pd


def printSongs(recommended_songs, liked_songs_trigger=False, title= "General"):
    recommended_songs = recommended_songs['track_name'] + ' - ' +  recommended_songs['artist_name']
    liked_songs = liked_songs_df['track_name'] + ' - ' +  liked_songs_df['artist_name']
    
    if liked_songs_trigger:
        print("\n")
        print("Your liked songs are: ")
        for i, song in enumerate(liked_songs):
            print(f"{i+1}. {song}")
    
    print("\n")
    print("The " + title + " songs are: " )
    for i, song in enumerate(recommended_songs):
        print(f"{i+1}. {song}")




if __name__ == '__main__':
    n_clusters = 10
    n_clusters_sentiment= 10
    n_clusters_year_genre = 10 
    k = 10
    liked_songs = []
    file_songs = "liked_songs.txt"  # Nombre del archivo
    
     
    # pop_dataset = pd.DataFrame()
    # rock_dataset = pd.DataFrame()
    # blues_dataset = pd.DataFrame()
    # country_dataset = pd.DataFrame()
    # jazz_dataset = pd.DataFrame()
    # hiphop_dataset = pd.DataFrame()
    # reggae_dataset = pd.DataFrame()
    
    # dataset_set = [pop_dataset, rock_dataset, blues_dataset, country_dataset, jazz_dataset, hiphop_dataset, reggae_dataset]
    
    # Leer el contenido actual del archivo
    with open(file_songs, "r") as f:
        liked_songs = set(f.read().splitlines())
    
   
    # LOAD DB
    music_db, popularity, artists = data.load_data()
    merged_ds = data.merge_datasets(music_db, popularity, artists) 
    
    #Analisis
    #music_db = music_db.sample(n=10000, random_state=1)
    
    # music_db = data.clean_lyrics(music_db)
    # music_db = data.sentiment_analysis(music_db)
    data.encoding_and_normalization(music_db)
    
    
    #dataset_set = data.classify_gender(merged_ds, pop, rock, blues, country, jazz, hiphop, reggae)
    
    # #database sentimentally clustered
    # music_db = cluster.sentiment_cluster_songs(music_db, n_clusters_sentiment)
    # music_db = cluster.cluster_songs_by_genre_year(music_db, n_clusters_year_genre)
    
    #music_db.to_csv('Data/DB_2.0.csv')
    #database clustered 
    liked_songs_df = data.create_liked_songs_df(liked_songs, merged_ds)
    liked_songs_analyzed = data.analyze_liked_songs(liked_songs, merged_ds, n_clusters, sentiment_plot=True, year_genre_plot=True)
    

    correlation = data.search_correlation(merged_ds)
    
    # merged_recommended_songs = algs.recommend_songs(music_db, liked_songs_df, plot_recommendations=True)
    # emotionally_recommended_songs = algs.recommend_songs(music_db, liked_songs, plot_recommendations=True, sentiment=True)
    # genre_year_recommended_songs = algs.recommend_songs(music_db, liked_songs, plot_recommendations=True, genre_year=True)
    # merged_emotion_genre_year = algs.mergeRecommendations(music_db, emotionally_recommended_songs, genre_year_recommended_songs)
    cosine_recommendation_songs = algs.cosine_sim_Recommendations(liked_songs_df, merged_ds, plot_recommendations=True)
    knn_recommendation_songs = algs.kNNRecommendation(merged_ds, liked_songs_df, k)
    
    recommended_songs = cosine_recommendation_songs
    
    liked_songs = set(recommended_songs['track_name'])
    
    # # Escribir los elementos del conjunto en un archivo de texto
    # with open('liked_songs.txt', 'a') as f:
    #     for track_name in liked_songs:
    #         f.write(track_name + '\n')
    
    # printSongs(emotionally_recommended_songs, title="Emotionally")
    # printSongs(genre_year_recommended_songs, title="Genre/Year")
    # printSongs(general_recommendation_songs, title="General")
    # printSongs(merged_emotion_genre_year, title="Merged Emotion-Genre/Year")
    # printSongs(merged_recommended_songs, liked_songs_trigger=True, title="Merged Recommendations")
    printSongs(knn_recommendation_songs, liked_songs_trigger=True, title="KNN recommendation")
    printSongs(cosine_recommendation_songs, title="Cosine Similarity recommendation")