
from django.contrib import admin
from django.urls import path, include
from analyze import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.regoPage, name='register'),
    path('login/', views.loginPage, name='login'),
    path('dash/', include('analyze.urls')),
    path('logout/', views.LogOutpage, name='logout'),
    path('hist/', views.historyPage, name='history'),
    path('enhance-report/', views.enhance_report, name='enhance_report'),
    path('segmentation/<str:folder_name>/', views.segmentation_page, name='segmentation_page'),
    path('get-segmentation/<str:folder_name>/', views.get_segmentation, name='get_segmentation'),
    
    
]