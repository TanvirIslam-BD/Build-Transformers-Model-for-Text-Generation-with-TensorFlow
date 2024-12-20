# ml_app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('train/', views.train, name='train'),
    path('generate-text/', views.generate_text, name='generate_text'),
]
