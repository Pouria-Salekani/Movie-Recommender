'''

To see how the ekman and emoRoberta model were processed, please go to 'computation.py'
To see how the accuracy testing was done, please go to 'training.py'

NONE OF THOSE SCRIPTS ABOVE WERE USED (IN VS CODE), THOSE FILES ARE THERE JUST TO SHOW YOU THE PROCESS AND THE ACCURACY TESTING

'''


from nrclex import NRCLex
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from core import emotion_model

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')

import django
django.setup()


from django.conf import settings

#stores movies
movies = []

def clean_up_df():
    #read file first
    # dir_path = os.path.abspath("D:/movie_proj/models_csvs")
    # csv_path = os.path.join(dir_path, "movies.csv")



    #first read the csv and do some cleaning
    #df = pd.read_csv(csv_path)

    file_ = open(os.path.join(settings.BASE_DIR, 'movies.csv'))
    df =  pd.read_csv(file_)

    #used for nrclex, emo, and ekman only
    for index, value in df['genres'].items():
        if not pd.isna(value):
            ex1 = value.split()
        if ex1:
            if len(ex1) > 1 and ex1[1] == 'am':
                continue
            else:
                df.loc[index, 'genres'] = 'I am feeling ' + ex1[0].lower()

    else:
        df.loc[index, 'genres'] = 'none'

    df['emotions'] = None

    return df


def read_user_emotion(text):
    emotion_labels = emotion_model(text)
    
    return emotion_labels


def recommend_movie(model_df, text):
    emotion_index = -1

    #the user text
    user_text = read_user_emotion(text)
    text = user_text[0]['label']

    #responsible for printing out titles
    temp_df = model_df

    #dtype: object, only these are needed
    model_df = model_df['title'] + ' ' +model_df['genres'] + ' ' +model_df['emotions']


    #below we will find the first instance of the user emotion...
    for i, value in enumerate(model_df):
        cur_value = model_df[i]

        if not pd.isna(value):
            emotion = cur_value[cur_value.rfind(' ')+1:]

        if emotion == text:
            emotion_index = i
            break

    
    vector = TfidfVectorizer()
    fit_vector = vector.fit_transform(model_df)

    similar = cosine_similarity(fit_vector)
    similarity_score = list(enumerate(similar[emotion_index]))
    #want the most relevant; highest score closest to 1.000
    sorted_score = sorted(similarity_score, key = lambda i:-i[1])


    #print out top 15 movies
    movie = 0
    for i, value in sorted_score:
        movies.append(temp_df.loc[i, 'title'])

        movie += 1
        if movie == 15:
            break


def nrclex_model(user_text):
    nrc_df = clean_up_df()

    for index, value in nrc_df['genres'].items():
        text = value

        if not pd.isna(value):
            text_obj = NRCLex(text)
            emotions = text_obj.affect_frequencies

            sorted_emotions = sorted(emotions.items(), key = lambda i:-i[1])[0][0]
            
            if sorted_emotions == 'positive':
                nrc_df.loc[index, 'emotions'] = 'excitement'
            else:
                nrc_df.loc[index, 'emotions'] = sorted_emotions


    recommend_movie(nrc_df, user_text)



def movie_emo_model(user_text):
    # dir_path = os.path.abspath("D:/movie_proj/models_csvs")
    # csv_path = os.path.join(dir_path, "movies.csv")

    #movie_emo_df = pd.read_csv('/movie_proj/models_csvs/movies.csv')
    #movie_emo_df = pd.read_csv(csv_path)

    file_ = open(os.path.join(settings.BASE_DIR, 'movies.csv'))
    movie_emo_df =  pd.read_csv(file_)

    for g in ['genres']:
        movie_emo_df[g] = movie_emo_df[g].fillna('')


    for index, value in movie_emo_df['genres'].items():
        ex = value.split()
        if ex:
            movie_emo_df.loc[index, 'genres'] = ex[0]


    movie_emo_df['emotios'] = None


    for index, value in movie_emo_df['genres'].items():
        if value == 'Action':
            movie_emo_df.loc[index, 'emotions'] = 'excitement'

        elif value == 'Thriller':
            movie_emo_df.loc[index, 'emotions'] = 'surprise curiosity'

        elif value == 'Horror':
            movie_emo_df.loc[index, 'emotions'] = 'fear'

        elif value == 'Adventure':
            movie_emo_df.loc[index, 'emotions'] = 'excitement joy'

        elif value == 'Comedy':
            movie_emo_df.loc[index, 'emotions'] = 'joy amusement'

        elif value == 'Romance':
            movie_emo_df.loc[index, 'emotions'] = 'love caring approval'

        elif value == 'Mystery':
            movie_emo_df.loc[index, 'emotions'] = 'curiosity'

        elif value == 'Drama':
            movie_emo_df.loc[index, 'emotions'] = 'excitement'

        elif value == 'Fantasy':
            movie_emo_df.loc[index, 'emotions'] = 'adventure'

        elif value == 'Science':
            movie_emo_df.loc[index, 'emotions'] = 'curiosity'

        elif value == 'War':
            movie_emo_df.loc[index, 'emotions'] = 'sadness grief remorse anger'

        elif value == 'Western':
            movie_emo_df.loc[index, 'emotions'] = 'neutral'

        elif value == 'Family':
            movie_emo_df.loc[index, 'emotions'] = 'caring joy'

        elif value == 'Animation':
            movie_emo_df.loc[index, 'emotions'] = 'joy'

        elif value == 'Documentary':
            movie_emo_df.loc[index, 'emotions'] = 'curiosity realization'

        elif value == 'Crime':
            movie_emo_df.loc[index, 'emotions'] = 'realization'

        elif value == 'Fiction':
            movie_emo_df.loc[index, 'emotions'] = 'excitement curiosity'

        elif value == 'Music':
            movie_emo_df.loc[index, 'emotions'] = 'love caring'

        elif value == 'History':
            movie_emo_df.loc[index, 'emotions'] = 'neutral'

    

    for index, value in movie_emo_df['emotions'].items():
        if pd.isna(value):
            movie_emo_df.loc[index, 'emotions'] = 'none'


    recommend_movie(movie_emo_df, user_text)



#already preprocessed data; so will read from csv
#to see how they get processed... go to 'computation.py'
def ekman_model(user_text):
    # dir_path = os.path.abspath("D:/movie_proj/models_csvs")
    # csv_path = os.path.join(dir_path, "ekman_test.csv")

    #ekman_df = pd.read_csv('/movie_proj/models_csvs/ekman_test.csv')
    #ekman_df = pd.read_csv(csv_path)

    file_ = open(os.path.join(settings.BASE_DIR, 'ekman_test.csv'))
    ekman_df =  pd.read_csv(file_)

    #removing nan's
    for index, value in ekman_df['genres'].items():
       if pd.isna(value):
          ekman_df.loc[index, 'genres'] = 'none'
          ekman_df.loc[index, 'emotions'] = 'none'


    recommend_movie(ekman_df, user_text)


#read from csv
#to see how they get processed... go to 'computation.py'
def emoroberta_model(user_text):
    # dir_path = os.path.abspath("D:/movie_proj/models_csvs")
    # csv_path = os.path.join(dir_path, "emo_test.csv")

    #emoroberta_df = pd.read_csv('/movie_proj/models_csvs/emo_test.csv')
    #emoroberta_df = pd.read_csv(csv_path)

    file_ = open(os.path.join(settings.BASE_DIR, 'emo_test.csv'))
    emoroberta_df =  pd.read_csv(file_)

    #removing nan's
    for index, value in emoroberta_df['genres'].items():
       if pd.isna(value):
          emoroberta_df.loc[index, 'genres'] = 'none'
          emoroberta_df.loc[index, 'emotions'] = 'none'

    recommend_movie(emoroberta_df, user_text)
