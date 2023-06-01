# -*- coding: utf-8 -*-
"""
Created on Fri May 26 15:50:37 2023

@author: DavidXanxez
"""
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go



def plot_clusters(dataframe, n_clusters=10, recommended_songs=None, liked_songs = None, sentiment=False, genre_year=False, title=None, graph_3D=False):
    #colors = plt.cm.get_cmap('tab10', n_clusters)
    
    if sentiment:
        plt.figure(figsize=(10,8))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.scatter(dataframe['positive_sentiment'], dataframe['negative_sentiment'])
        
    
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
        
        plt.title(title, fontsize=12)
        
        plt.xlabel('positive_sentiment', fontsize=10)
        plt.ylabel('negative_sentiment', fontsize=10)


        plt.legend(fontsize=8)
        plt.show()
        
        # if graph_3D:
        #     # Crear figura y agregar un gráfico de dispersión 3D
        #     fig = go.Figure(data=go.Scatter3d(
        #         x=dataframe['positive_sentiment'],
        #         y=dataframe['negative_sentiment'],
        #         z=dataframe['neutral_sentiment'],
        #         mode='markers'
        #     ))
            
        #     # Ajustar opciones de diseño
        #     fig.update_layout(
        #         scene=dict(
        #             xaxis=dict(title='Positive Sentiment'),
        #             yaxis=dict(title='Negative Sentiment'),
        #             zaxis=dict(title='Neutral Sentiment')
        #         )
        #     )
            
        #     # Mostrar el gráfico interactivo en el navegador
        #     fig.show()

            
    if genre_year:
        
        plt.figure(figsize=(10,8))
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        
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

        plt.title(title, fontsize=12)
        plt.xlabel('genre', fontsize=10)
        plt.ylabel('release_date', fontsize=10)
        plt.legend(fontsize=8)
        plt.show()

    
        
        

