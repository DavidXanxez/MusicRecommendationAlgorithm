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
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt






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
    k = 20
    liked_songs = []
    liked_songs_txt = []
    file_songs = "liked_songs.txt"  # Nombre del archivo
    good_rec_file = "good_recommendation.txt"
    good_recommendations = []
    
    # Leer el contenido actual del archivo
    with open(file_songs, "r") as f:
        liked_songs_txt = set(f.read().splitlines())
    
    # Leer el contenido actual del archivo
    with open(good_rec_file, "r") as f:
        good_recommendations = set(f.read().splitlines())
    
   
    # LOAD DB
    music_db, popularity, artists, user_db = data.load_data()
    merged_ds = data.merge_datasets(music_db, popularity, artists) 
    
    
    music_db = data.clean_lyrics(merged_ds)
    music_db = data.sentiment_analysis(merged_ds)
    data.encoding_and_normalization(merged_ds)
    
    
    
    
    
    
    #######################################################USER-CONTENT MATRIX##########################################################
    
    recommendation_matrix = pd.read_csv(r"C:\Users\34653\Documents\UPF\TFG\MusicRecommendationAlgorithm\Code\recomendaciones.csv", sep=' ', dtype=int)
    df = recommendation_matrix.astype(int)
    
    # Utiliza la matriz cargada y convertida en tu código
    recommendation_matrix = df.values
        
    #recommendation_matrix = data.createRecommendationMatrix(merged_ds, user_db)
    
    

    canciones_por_usuario = {}
    
    for index in range(len(recommendation_matrix)):
        # Obtiene el nombre del usuario
        usuario = f'Usuario {index + 1}'
        
        # Obtiene las canciones que le gustan al usuario (valores mayores a 0)
        liked_songs = recommendation_matrix[index][recommendation_matrix[index] > 0].tolist()
        
        # Guarda las canciones en el diccionario
        canciones_por_usuario[usuario] = liked_songs
        
    # # Imprime las canciones por usuarios
    # for usuario, canciones in canciones_por_usuario.items():
    #     print(usuario + ":")
    #     for cancion in canciones:
    #         print("- " + cancion)
    #     print()

    canciones_unicas = merged_ds['track_name'].unique()
    
    usuario_id = 367
    
    # Extraer las preferencias del usuario de la matriz de recomendación
    indices_likes = np.where(recommendation_matrix[usuario_id] == 1)[0]
    
    # Obtener los nombres de las canciones correspondientes a las posiciones encontradas
    canciones_gustan = [canciones_unicas[idx] for idx in indices_likes]
    
    # Dividir en conjunto de entrenamiento (80%) y conjunto de prueba (20%)
    train_preferences, test_preferences = train_test_split(canciones_gustan, test_size=0.2, train_size=0.8, random_state=43)
    

    train_preferences = data.create_song_attributes_df(train_preferences, merged_ds).head(100)
    #test_preferences = data.create_liked_songs_df(test_preferences, merged_ds)

    train_predictions = algs.kNNRecommendation(merged_ds, train_preferences, k)
    #test_predictions = algs.kNNRecommendation(merged_ds, test_preferences, k)
    
    train_predictions = data.create_song_attributes_df(train_preferences, merged_ds).head(100)
    #test_predictions = data.create_liked_songs_df(test_predictions, merged_ds)
    
    
    columnas_seleccionadas = ['positive_sentiment', 'negative_sentiment', 'neutral_sentiment', 'violence', 'loudness', 'acousticness', 'energy']
    
    train_precision_scores = []
    
    ##########################################PRECISION;RECALL;F!;ROCAUC####################################################
    # for columna in columnas_seleccionadas:
    #     # Obtener los valores reales y predichos de la columna seleccionada
    #     valores_reales = train_preferences[columna]
    #     valores_predichos = train_predictions[columna]
    
    #     # Calcular la precisión para la columna actual
    #     precision = precision_score(valores_reales, valores_predichos)
    
    #     # Agregar el resultado a la lista de puntuaciones de precisión
    #     train_precision_scores.append(precision)
    
    # # Imprimir las puntuaciones de precisión por columna
    # for columna, precision_marks in zip(columnas_seleccionadas, train_precision_scores):
    #     print(f'Precisión para {columna}: {precision_marks}')
    #########################################################################################################################
  # Crear listas para almacenar los valores de los errores
    mae_scores = []
    mse_scores = []
    r2_scores = []
    
    for columna in columnas_seleccionadas:
        valores_reales = train_preferences[columna]
        valores_predichos = train_predictions[columna]
    
        # Calcular los errores
        mae = mean_absolute_error(valores_reales, valores_predichos)
        mse = mean_squared_error(valores_reales, valores_predichos)
        r2 = r2_score(valores_reales, valores_predichos)
    
        # Agregar los valores a las listas
        mae_scores.append(mae)
        mse_scores.append(mse)
        r2_scores.append(r2)
    
    # Crear el gráfico de barras
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(columnas_seleccionadas))
    
    # Configurar las barras
    bar_width = 0.25
    opacity = 0.8
    
    rects1 = ax.bar(x, mae_scores, bar_width, alpha=opacity, color='b', label='MAE')
    rects2 = ax.bar([i + bar_width for i in x], mse_scores, bar_width, alpha=opacity, color='g', label='MSE')
    rects3 = ax.bar([i + 2 * bar_width for i in x], r2_scores, bar_width, alpha=opacity, color='r', label='R2')
    
    # Configurar los ejes y etiquetas
    ax.set_xlabel('Columnas')
    ax.set_ylabel('Error')
    ax.set_title('Errores MAE, MSE y R2')
    ax.set_xticks([i + bar_width for i in x])
    ax.set_xticklabels(columnas_seleccionadas)
    ax.legend()
    
    # Mostrar el gráfico
    plt.tight_layout()
    plt.show()
    
    
    
    
    
    # # Calcular las métricas
    # train_precision = precision_score(train_preferences, train_predictions)
    # train_recall = recall_score(train_preferences, train_predictions)
    # train_f1 = f1_score(train_preferences, train_predictions)
    # train_roc_auc = roc_auc_score(train_preferences, train_predictions)
    
    # test_precision = precision_score(test_preferences, test_predictions)
    # test_recall = recall_score(test_preferences, test_predictions)
    # test_f1 = f1_score(test_preferences, test_predictions)
    # test_roc_auc = roc_auc_score(test_preferences, test_predictions)
    
    ####################################################################################################################################
    #######################################USER BASED FILTERING#########################################################################
    # recommendation_matrix = np.genfromtxt('recomendaciones.csv', delimiter=' ', dtype=int)
    
    # # Crea un diccionario para almacenar las canciones por usuarios
    # canciones_por_usuario = {}
    
    
    # df_usuarios = pd.DataFrame({'nombre': [f'Usuario {i+1}' for i in range(len(recommendation_matrix))]})
    # df_canciones = pd.DataFrame({'nombre': canciones_unicas})
    
    # # Crear un DataFrame para almacenar las canciones que le gustan a cada usuario
    # df_canciones_usuario = pd.DataFrame(columns=['usuario', 'cancion'])
    # for i, row in enumerate(recommendation_matrix):
    #     usuario = f'Usuario {i+1}'
    #     liked_songs = [canciones_unicas[j] for j, song_rating in enumerate(row) if song_rating > 0]
    #     df_user_songs = pd.DataFrame({'usuario': usuario, 'cancion': liked_songs})
    #     df_canciones_usuario = pd.concat([df_canciones_usuario, df_user_songs], ignore_index=True)
    
    
    
        
    # # Agrupar los datos por usuario y obtener las canciones como listas
    # canciones_por_usuario = df_canciones_usuario.groupby('usuario')['cancion'].apply(list)
    

    
    # resultados_por_usuario = {}
    
    # primeros_usuarios = canciones_por_usuario.head(5)
    
    # # Iterar sobre las listas de canciones de cada usuario
    # for usuario, canciones in primeros_usuarios.items():
    #     # Pasar la lista de canciones a la función deseada y guardar el resultado
    #     liked_songs_df = data.create_liked_songs_df(canciones, merged_ds)
    #     liked_songs_analyzed = data.analyze_liked_songs(canciones, merged_ds, sentiment_plot=True, year_genre_plot=True)
        
    #     #print(liked_songs_analyzed)
        
    #     knn_recommendation_songs_per_user = algs.kNNRecommendation(merged_ds, liked_songs_df, k)
    #     resultados_por_usuario[usuario] = knn_recommendation_songs_per_user 

    #################################################################################################################################
    
    # #dataset_set = data.classify_gender(merged_ds, pop, rock, blues, country, jazz, hiphop, reggae)
    
    # # #database sentimentally clustered
    # # music_db = cluster.sentiment_cluster_songs(music_db, n_clusters_sentiment)
    # # music_db = cluster.cluster_songs_by_genre_year(music_db, n_clusters_year_genre)
    
    # #music_db.to_csv('Data/DB_2.0.csv')
    # #database clustered 
    liked_songs_df = data.create_liked_songs_df(liked_songs_txt, merged_ds)
    #liked_songs_analyzed = data.analyze_liked_songs(liked_songs, merged_ds, n_clusters, sentiment_plot=True, year_genre_plot=True)
    

    # correlation = data.search_correlation(merged_ds)
    
    # # merged_recommended_songs = algs.recommend_songs(music_db, liked_songs_df, plot_recommendations=True)
    # # emotionally_recommended_songs = algs.recommend_songs(music_db, liked_songs, plot_recommendations=True, sentiment=True)
    # # genre_year_recommended_songs = algs.recommend_songs(music_db, liked_songs, plot_recommendations=True, genre_year=True)
    # # merged_emotion_genre_year = algs.mergeRecommendations(music_db, emotionally_recommended_songs, genre_year_recommended_songs)
    cosine_recommendation_songs = algs.cosine_sim_Recommendations(liked_songs_df, merged_ds, plot_recommendations=True)
    #knn_recommendation_songs = algs.kNNRecommendation(merged_ds, liked_songs_df, k)
    
    # recommended_songs = cosine_recommendation_songs
    
    # liked_songs = set(recommended_songs['track_name'])
    
    # # # Escribir los elementos del conjunto en un archivo de texto
    # # with open('liked_songs.txt', 'a') as f:
    # #     for track_name in liked_songs:
    # #         f.write(track_name + '\n')
    
    # # printSongs(emotionally_recommended_songs, title="Emotionally")
    # # printSongs(genre_year_recommended_songs, title="Genre/Year")
    # # printSongs(general_recommendation_songs, title="General")
    # # printSongs(merged_emotion_genre_year, title="Merged Emotion-Genre/Year")
    # # printSongs(merged_recommended_songs, liked_songs_trigger=True, title="Merged Recommendations")
    #printSongs(knn_recommendation_songs, liked_songs_trigger=True, title="KNN recommendation")
    # printSongs(cosine_recommendation_songs, title="Cosine Similarity recommendation")