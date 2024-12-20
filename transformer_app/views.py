from django.http import JsonResponse
from django.shortcuts import render

from transformer_app.transformer.train import train_model, generate_text_with_transformer


def index(request):
    return render(request, 'index.html')

def train(request):
    result = train_model()
    return JsonResponse(result)

def generate_text(request):
    result = generate_text_with_transformer()
    return JsonResponse(result)

