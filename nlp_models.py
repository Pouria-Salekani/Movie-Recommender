
#this is where all the code will go

#4 separate methods for each model (once done, each method calls the recommend_movie())
#1 method for finding the movie cosine and stuff

#then return to views.py by importing this class
from nrclex import NRCLex
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#first read the csv and do some cleaning
df = pd.read_csv('/movie_proj/models_csvs/movies.csv')

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


def recommend_movie(model_df, text):
    text = text[text.rfind(' ')+1:]

    #dtype: object, only these r needed
    model_df = model_df['title'] + ' ' +model_df['genres'] + ' ' +model_df['emotions']

    #below we will find the first instance of the user emotion...
    
    vector = TfidfVectorizer()
    fit_vector = vector.fit_transform(model_df)

    similar = cosine_similarity(fit_vector)
    similarity_score = list(enumerate(similar[index]))


# #gets the text from user, cleans it up, and passes it to 'recommend_movies()'
# def user_emotion(text):
#     text = 'I am feeling happy'

#     text = text[text.rfind(' ')+1:]
#     #print(text)

#     recommend_movie(None, text)



user_emotion('lo')


def nrclex_model(user_text):
    nrc_df = df

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
    movie_emo_df = pd.read_csv('/movie_proj/models_csvs/movies.csv')

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
    ekman_df = pd.read_csv('/movie_proj/models_csvs/ekman_test.csv')

    #removing nan's
    for index, value in ekman_df['genres'].items():
       if pd.isna(value):
          ekman_df.loc[index, 'genres'] = 'none'
          ekman_df.loc[index, 'emotions'] = 'none'


    recommend_movie(ekman_df, user_text)


#read from csv
#to see how they get processed... go to 'computation.py'
def emoroberta_model(user_text):
    emoroberta_df = pd.read_csv('/movie_proj/models_csvs/emo_test.csv')

    #removing nan's
    for index, value in emoroberta_df['genres'].items():
       if pd.isna(value):
          emoroberta_df.loc[index, 'genres'] = 'none'
          emoroberta_df.loc[index, 'emotions'] = 'none'

    recommend_movie(emoroberta_df, user_text)



# def recommend_movie(model_df=None, text=None):
#     print(text)
#     pass



# recommend_movie()


