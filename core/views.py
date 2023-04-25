from django.shortcuts import render, redirect


def home(request):
    return render(request, 'core/home.html')



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


    return render(request, 'core/search.html')



def results(request):
    movies = request.session.get('movies')
    return render(request, 'core/results.html', {'movies':movies})

