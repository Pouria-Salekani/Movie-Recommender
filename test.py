import pandas as pd

df = pd.read_csv('models_csvs/final_results.csv')
df = df.drop('Unnamed: 0', axis=1)

print(df)



#---------similar layout to the amazon wilson score
#user types in emotion
    #user then has an option to pick which model it wants to use (default being self since most accurate)
#django reads for post
#then uses the emoroberta to read it
#then uses cosine similarity and tdidf vectorization 
#then displays top 15 movies