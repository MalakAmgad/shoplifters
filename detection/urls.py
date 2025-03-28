from django.urls import path
from .views import index, detect

urlpatterns = [
    path("", index, name="home"),
    path("detect/", detect, name="detect"),
]
