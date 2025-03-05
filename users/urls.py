from django.urls import path
from django.contrib.auth import views as auth_views

from users.views import HtmxLogoutView

urlpatterns = [
    path(
        "login/",
        auth_views.LoginView.as_view(
            template_name="users/login.html", redirect_authenticated_user=True
        ),
        name="login",
    ),
    path("logout/", HtmxLogoutView.as_view(), name="logout"),
]
