# Movie-Recommender

A website made using Python & Django that uses NLP (Natural Language Processing) to recommend movies based off of the emotion you are *currently feeling*.

I made this project because I wanted try using some HuggingFace models and wanted to learn some NLP skills to showoff.

***This project uses [movies.csv](https://github.com/Pouria-Salekani/Movie-Recommender/files/11326317/movies.csv) as the dataset, so, not ALL movies you get recommended are up-to-date. I think the movie year ranges are [1985, ~2018]***


## Table of Contents

- [Website](#website)
- [Instructions](#instructions)
- [Algorithms](#algorithms)
  - [My Algorithm](#my-algorithm)
    - [Alternative](#alternative)
  - [EmoRoberta](#emoroberta)
  - [Ekman](#ekman)
  - [NRCLex](#nrclex)
- [Training and Testing](#training-and-testing)
  - [Measuring accuracy](#measuring-accuracy) 
  - [Accuracy Score](#accuracy-score)
- [LIMITATIONS](#limitations)
- [Computation](#computation)
- [Movie Dataset](#movie-dataset)
- [Credits and References](#credits-and-references)



## Website

Website is here: https://web-production-2cb2.up.railway.app/

## Instructions

In the text-box, type out how you are currently feeling in this format: ***"I am feeling x"*** where ***"x"*** is your current emotion.
* **Example**: `I am feeling happy`
* **Example**: `I am feeling scared` 


**DISCLAIMER**: Obviously there are some limitations and the program is not perfect, please go to the ***[limitations](#limitations)*** section for more info. 


## Algorithms

This program uses *4 different algorithms/models* to recommend you movies. First being an algorithm I made, the second & third being a model from [HuggingFace.com](https://huggingface.co/), last one, *"NRCLex"* which ***isn't a model***, rather, a it is a Python package that provides a lexicon-based approach for natural language processing tasks, such as sentiment analysis, emotion detection, and opinion mining... however, for sake of this project, I will refer ***NRCLex*** as a *model*.


### My Algorithm

This is an algorithm that I made using the dataset provided above in the introduction. This algorithm reads the *genres* of the movies and tries to assign it common emotions that is experienced by the watcher. (**obviously these emotions are subjective**)
<br>
Below I will show the code of how this algorithm was constructed. 

```python

def movie_emo_model(user_text):
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

    
    #this is because some GENRES were left blank in the .csv file
    for index, value in movie_emo_df['emotions'].items():
        if pd.isna(value):
            movie_emo_df.loc[index, 'emotions'] = 'none'

```

The "emotions" assigned to those "genres" are not random. Those emotions coincides with the other 3 models that I will describe below. If you want to see the accuracy score of each mode, please go to the *[Accuracy Score](#accuracy-score)* section.


#### Alternative

I chose to construct my algorithm like that because the algorithm works very well with the [EmoRoberta](#emoroberta), *again, you can see the accuracy scores here: [Accuracy Score](#accuracy-score)*
<br>

I could have used the `NLTK Python` library and maybe `TextBlob`, however, after tesing it out in **Google Colabs** the results were tremendously terrible. Moreover, using the `NLTK` to read *genres* would mostly return some "neutral" value or **neutral** itself.
<br>

In addition, I thought of reading the ***movie description*** which is around 2-3 sentences and conjure up an emotion based off of that, but that did not work either. I tried using stopwords, lemmatizing, and stemming, and more. Another thing to note, *"we	 did	 not	 use	 stemming	 of	 words	 as	 some	
information is	lost while	stemming	a word	to	its	root	form"* (reference: https://cseweb.ucsd.edu/classes/wi15/cse255-a/reports/fa15/003.pdf)


### EmoRoberta

This is a model from https://huggingface.co/arpanghoshal/EmoRoBERTa, this model was trained with a *"Dataset labelled 58000 Reddit comments with 28 emotions"*. If you visit the website, you can see that the emotions from the ***[My Algorithm](#my-algorithm)*** match together.

<br>

This is also the same model that is used to read `user_text`.

### Ekman

This is a model from https://huggingface.co/arpanghoshal/EkmanClassifier, this model is very similar to the ***[NRCLex](#nrclex)*** since they both use **"six basic emotions"**, however, NRCLex also comes with *positive* and *negative* sentiments, which can be used for **Sentiment Analysis**.


### NRCLex

Similar to the [Ekman](#ekman), this library *"contains approximately 27,000 words, and is based on the National Research Council Canada (NRC) affect lexicon"*. They do give similar resultss, however, the [Ekman](#ekman) is **much more accurate**


## Training and Testing

After putting together all the 4 options, I then started training and testing them so I can come up with an *accuracy score*. The training and testing was done with the `scikit-learn` module.

### Measuring Accuracy

Below is Python code to show how the accuracy was measured. **You can also find the entire code in `training.py` script.


```python

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
labels = LabelEncoder()


#these were respective dataframes that was already compiled and processed in Google Colab...
#You can also find the dataframes in 'nlp_models.py', NRCLEX and movie_alg have their own processed dataframe there
movie_df = None
lex_df = None


emo = pd.read_csv('/content/emo_test.csv')
ekkman = pd.read_csv('/content/ekman_test.csv')
movie_alg = movie_df
lexy = lex_df


#removing nan's from the dataframes because *SOME* movies DO NOT have a 'genre'
def remove_nan():
    for index, value in emo['emotions'].items():
        if pd.isna(value):
            emo.loc[index, 'emotions'] = 'placeholder'

    for index, value in ekkman['emotions'].items():
        if pd.isna(value):
            ekkman.loc[index, 'emotions'] = 'placeholder'

    for index, value in lexy['emotions'].items():
        if pd.isna(value):
            lexy.loc[index, 'emotions'] = 'placeholder'


#this is where the accuracy testing is done
def accuracy():

    #assume that movie_orig is the original 'y' i.e. the results we want for the models to predict
    movie_orig = None

    emo_X = emo['emotions']
    ekman_X = ekkman['emotions']
    movie_X = movie_alg['emotions'] 
    lexy_X = lexy['emotions']

    y = movie_orig['emotions'] 


    plots = {} #key = current alg | value = hashmap of models as keys and scores as values


    for key, current_X in [('emoRoberta', emo_X), ('ekman', ekman_X), ('movie_alg', movie_X),  ('nrclex', lexy_X)]:
        #need to turn output (__X) into numerical representation
        X_labeling = labels.fit_transform(current_X)
        X = X_labeling.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

        model_tree = DecisionTreeClassifier()
        model_log = LogisticRegression()
        model_forest = RandomForestClassifier()

        # #fit them first
        model_tree.fit(X_test, y_test) 
        model_log.fit(X_test, y_test)
        model_forest.fit(X_test, y_test)

        #Decision Tree
        predict_tree = model_tree.predict(X_test)
        score_tree = accuracy_score(y_test, predict_tree)

        #Logistic Regression
        predict_log = model_log.predict(X_test)
        score_log = accuracy_score(y_test, predict_log)

        #Random Forest
        predict_forest = model_forest.predict(X_test)
        score_forest = accuracy_score(y_test, predict_forest)

        plots[key] = {'Decision Tree': score_tree, 'Logistic Regression': score_log, 'Random Forest': score_forest}


    return plots

```

Some things to note:

* The ***input*** ("X") were emotions constructed using the *models* , there were NOT numerical, so I used `LabelEncoder()` to convert the *emotions* into numerical representations.
* The ***output*** ("y") was the emotions that the models were trying to predict.
* The `plots` dictionary was used with `seaborn` to make the graph below.


### Accuracy Score

Below is an image to show the accuracy score for all 4 models.

![download](https://user-images.githubusercontent.com/27398502/234398026-7a3da3d0-7ed7-4997-8ebe-456723f9d091.png)














