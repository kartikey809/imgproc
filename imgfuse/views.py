from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets
from imgfuse.serializers import UploadSerializer
from imgfuse.models import Upload
from django.http import HttpResponse
import json

class UploadViewSet(viewsets.ModelViewSet):
   queryset = Upload.objects.all()
   serializer_class = UploadSerializer
   def post(self, request, *args, **kwargs):
        file = request.data['file']
        image = Upload.objects.create(image=file)
        return HttpResponse(json.dumps({'message': "Uploaded"}), status=200)