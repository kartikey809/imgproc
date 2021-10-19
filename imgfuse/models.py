from django.db import models
from .utils import *
from PIL import Image
import pywt
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from io import BytesIO
from django.core.files.base import ContentFile
# Create your models here.
ACTION_CHOICES = (
    ('meanmean','MEANMEAN'),
    ('meanmax','MEANMAX'),
    ('meanmin','MEANMIN'),
    ('maxmean','MAXMEAN'),
    ('maxmax','MAXMAX'),
    ('maxmin','MAXMIN'),
    ('minmean','MINMEAN'),
    ('minmax','MINMAX'),
    ('minmin','MINMIN'),

)

class Upload(models.Model):
    img1 = models.ImageField(upload_to='images')
    img2 = models.ImageField(upload_to='images')
    Fused_img = models.ImageField(upload_to='images',blank=True)
    FUSION_METHOD = models.CharField(max_length=50, choices=ACTION_CHOICES)

    def __str__(self):
        return str(self.id)
    def save(self, *args, **kwargs):
        pil_img1 = Image.open(self.img1)
        cv_img1 = np.array(pil_img1)
        pil_img2 = Image.open(self.img2)
        cv_img2 = np.array(pil_img2)
        pil_img = Image.open(self.Fused_img)
        img = fusion(cv_img1,cv_img2,FUSION_METHOD=self.FUSION_METHOD)
        im_pil = Image.fromarray(img)
        buffer = BytesIO()
        im_pil.save(buffer,format='png')
        image_png = buffer.getvalue()
        self.Fused_img.save(str(self.Fused_img),ContentFile(image_png),save=False)
        super().save()