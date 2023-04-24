'''

Below is the code to demonstrate how the accuracy was measured (the resulting image is on GitHub and the "models_csvs/final_results.csv" contains the final results of the test)
The code below was first tested in Google Colab then brought here


'''

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



#this is how the resulting image on GitHub was made
def make_image():
    import seaborn as sns

    results = accuracy()
    bv = pd.DataFrame(results).T

    df = pd.DataFrame(bv)
    df = df.set_index('Models')

    sns.heatmap(df, annot=True, cmap='OrRd')
