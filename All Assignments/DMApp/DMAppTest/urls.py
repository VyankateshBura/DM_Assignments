from django.urls import path,include

from . import views

urlpatterns = [
    path('hits/',views.HITSAlgo,name='hits')
]