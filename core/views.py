from django.shortcuts import render, redirect
import pandas as pd
import json


def home(request):

    df = pd.read_csv('models_csvs/final_results.csv')
    df = df.drop('Unnamed: 0', axis=1)

    table_html = df.to_html()

    if request.method == 'POST':
        print('form submitted')


    return render(request, 'home.html', {'table_html':table_html})



def search(request):
    #1 default
    #2 ekman
    #3 emo
    #4 nrclex
    request.session.flush()
    option = request.GET.get('option')

    print('OPTION PICKED = ', option)

    if request.method == 'POST':
        user_text = request.POST.get('emotion_text')

        #when we import it, it automatically runs the entire script, default python stuff
        import nlp_models

        #gets rid of previous data
        movies = nlp_models.movies
        movies.clear()

        if int(option) == 1:
            nlp_models.movie_emo_model(user_text)
            #movies = nlp_models.movies
            #request.session.flush()
            request.session['movies'] = movies
            return redirect('results')
            #print(nlp_models.movies)

        elif int(option) == 2:
            nlp_models.ekman_model(user_text)
            request.session['movies'] = movies
            return redirect('results')

        elif int(option) == 3:
            nlp_models.emoroberta_model(user_text)
            request.session['movies'] = movies
            return redirect('results')

        elif int(option) == 4:
            nlp_models.nrclex_model(user_text)
            request.session['movies'] = movies
            return redirect('results')


    return render(request, 'search.html')



def results(request):
    #items_json = request.GET.get('items')
    #movies = json.loads(items_json)
    movies = request.session.get('movies')
    print('MOVIESSSSSSS ', movies)
    return render(request, 'results.html', {'movies':movies})

#TODO: install SASS onto system