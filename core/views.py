from django.shortcuts import render
import pandas as pd


def home(request):

    df = pd.read_csv('models_csvs/final_results.csv')
    df = df.drop('Unnamed: 0', axis=1)

    table_html = df.to_html()

    if request.method == 'POST':
        print('form submitted')


    return render(request, 'home.html', {'table_html':table_html})



def search(request):
    option = request.GET.get('option')

    print('OPTION PICKED = ', option)

    if request.method == 'POST':
        print('form submitted')


    return render(request, 'search.html')

#TODO: install SASS onto system