"""
URL configuration for strvsocial project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.urls import path

from core.views import (
    HomeView,
    ProfileDetailView,
    ContentPostDetailView,
    CreateContentPostView,
    GetSimilarPostsHtmxView,
)

urlpatterns = [
    path("", HomeView.as_view(), name="home"),
    path("u/<str:username>/", ProfileDetailView.as_view(), name="profile-detail"),
    path("p/<uuid:uuid>/", ContentPostDetailView.as_view(), name="content-post-detail"),
    path("p/create/", CreateContentPostView.as_view(), name="create-content-post"),
    path(
        "p/<uuid:uuid>/similar-posts/",
        GetSimilarPostsHtmxView.as_view(),
        name="get-similar-posts",
    ),
]
