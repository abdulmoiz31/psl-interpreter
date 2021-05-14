from django.shortcuts import render
from django.http import JsonResponse
from testing.models import *
from .api.serializers import *
import requests
# Create your views here.

    
def add_data(request):
    url = "https://covid-19-tracking.p.rapidapi.com/v1"
    headers = {
    'x-rapidapi-key': "aaba10cb3emshc19de4d004614eap156767jsn622918633cd4",
    'x-rapidapi-host': "covid-19-tracking.p.rapidapi.com"
    }
    response = requests.request("GET", url, headers=headers)
    #Total Cases_text
    #Country_text
    #Total Deaths_text
    #Total Recovered_text
    json_response = response.json()
    #Country.objects.get(name="World")
    for x in range(len(json_response)-1):
        Country.objects.create(name=json_response[x]['Country_text'],cases=json_response[x]['Total Cases_text'],
        deaths=json_response[x]['Total Deaths_text'],recovered=json_response[x]['Total Recovered_text'])
    #allObj = Country.objects.all()
    #for x in range(len(allObj)):
    #    a = allObj[x]
    #    a.delete()
    return JsonResponse("records added",safe=False)
