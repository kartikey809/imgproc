from rest_framework import serializers

from imgfuse.models import Upload

class UploadSerializer(serializers.ModelSerializer):
   class Meta:
       model = Upload
       fields = ('__all__')


