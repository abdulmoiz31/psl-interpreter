from django.urls import path
from .views import * 

urlpatterns = [
    path("<pk>/",CountryView.as_view()),
    path("test",CountryView.as_view()),
    #path(r'^upload/(?P<filename>[^/]+)$', CountryView.as_view())
]