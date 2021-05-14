from django.db import models

# Create your models here.
class Country(models.Model):
    name = models.CharField(max_length=60)
    cases = models.CharField(max_length=60)
    deaths = models.CharField(max_length=60)
    recovered = models.CharField(max_length=60)
