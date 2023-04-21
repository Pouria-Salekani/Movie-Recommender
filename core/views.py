from django.shortcuts import render
import pandas as pd


def home(request):

    df = pd.read_csv('models_csvs/final_results.csv')
    df = df.drop('Unnamed: 0', axis=1)

    table_html = df.to_html()
    return render(request, 'core/home.html', {'table_html':table_html})