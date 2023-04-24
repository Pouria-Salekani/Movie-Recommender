'''

This file will NOT be used, this will only showcase how the emoRoberta and Ekman processed the 'movies.csv'
All of this code was tested in Google Colab then brought here to VS Code


'''


import pandas as pd
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline


#first convert each genre into a 'feeling' (also called it 'cleaning up data')
def convert_to_feeling():
    df = pd.read_csv('/content/movies.csv')

    for g in df['genres']:
        if not g:
            df[g] = df[g].fillna('')


    for index, value in df['genres'].items():
        if not pd.isna(value):
            ex1 = value.split()
            if ex1:
                if len(ex1) > 1 and ex1[1] == 'am':
                    continue
                else:
                    df.loc[index, 'genres'] = 'I am feeling ' + ex1[0].lower()


    df['emotions'] = None


    return df


def emoroberta_processing():
    lvg = convert_to_feeling()

    tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
    model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoRoBERTa")

    emotion = pipeline('sentiment-analysis', 
                        model='arpanghoshal/EmoRoBERTa')
    
    for index, value in lvg['genres'].items():
        if not pd.isna(value):
            emotion_labels = emotion(value)
            emotionz = emotion_labels[0]['label']
            lvg.loc[index, 'emotions'] = emotionz

    #this is the same .csv in this project
    lvg.to_csv('emo_test.csv', index=False)


def ekman_processing():
    bhj = convert_to_feeling()

    ekman = pipeline('sentiment-analysis', model='arpanghoshal/EkmanClassifier')
    for index, value in bhj['genres'].items():
        if not pd.isna(value):
            ekman_labels = ekman(value)
            emotions = ekman_labels[0]['label']
            bhj.loc[index, 'emotions'] = emotions

    #this is the same .csv in this project
    bhj.to_csv('ekman_test.csv', index=False)



