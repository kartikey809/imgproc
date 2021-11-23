from django.urls import include, path

from rest_framework import routers, views

from imgfuse.views import UploadViewSet

router = routers.DefaultRouter()
router.register(r'upload', UploadViewSet)

urlpatterns = [
   path('', include(router.urls)),

]