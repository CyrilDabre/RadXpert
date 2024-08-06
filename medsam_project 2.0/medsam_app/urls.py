# urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.upload_image, name='upload_image'),
    path('segmentation/<str:folder_name>/', views.segmentation_page, name='segmentation_page'),
    path('get-segmentation/<str:folder_name>/', views.get_segmentation, name='get_segmentation'),
]
