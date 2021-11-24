from django.db import models
# from .utils import *
from PIL import Image
import pywt
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
from io import BytesIO
from django.core.files.base import ContentFile

from imgfuse.utils import fusion,comp
# Create your models here.

#find the mean value in the array 
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

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
USES = (
    ('IMAGE RESTORATION','image restoration'),
    ('FACE MORPHING','face morphing'),
    ('IMAGE MIXING','image mixing'),
)

class Upload(models.Model):
    img1 = models.ImageField(upload_to='images')
    img2 = models.ImageField(upload_to='images')
    USE = models.CharField(max_length=255,choices= USES)
    Fused_img = models.ImageField(upload_to='images',blank=True)
    Expected_img = models.ImageField(upload_to='images',null=True,blank=True)
    FUSION_METHOD = models.CharField(max_length=50, choices=ACTION_CHOICES,blank=True)

    def __str__(self):
        return str(self.id)
    def save(self, *args, **kwargs):
        pil_img1 = Image.open(self.img1)
        cv_img1 = np.array(pil_img1)
        pil_img2 = Image.open(self.img2)
        cv_img2 = np.array(pil_img2)
        pil_img = Image.open(self.Fused_img)

        Expected_img =Image.open(self.Expected_img)
        cv_exp =np.array(Expected_img)
        arr=[]
        arr_of_s = []
        cv_exp = cv2.cvtColor(cv_exp, cv2.COLOR_BGR2GRAY)
        for choice in ACTION_CHOICES:
            fs = fusion(cv_img1,cv_img2,FUSION_METHOD=choice[0])
            fs = cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY)
            res, val = comp(fs,cv_exp)
            arr.append(res)
            arr_of_s.append(val)
        mean_of_s = sum(arr_of_s) / len(arr_of_s)
        print("The value closest to the mean of the array is: ")
        print(find_nearest(arr_of_s, mean_of_s))
        max_index = 0
        print(arr_of_s)
        print(arr)
        for i in range (1,len(arr)):
            if(arr[max_index]<arr[i]):
                max_index = i
        best = ACTION_CHOICES[max_index][0]
        print("________best method _________")
        print(best)
        img = fusion(cv_img1,cv_img2,FUSION_METHOD=best)
        im_pil = Image.fromarray(img)
        buffer = BytesIO()
        im_pil.save(buffer,format='png')
        image_png = buffer.getvalue()
        self.Fused_img.save(str(self.Fused_img),ContentFile(image_png),save=False)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # cv_exp = cv2.cvtColor(cv_exp, cv2.COLOR_BGR2GRAY)
        res = comp(img,cv_exp)
        print(res)
        super().save()

