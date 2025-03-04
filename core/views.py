from django.contrib.auth import get_user_model
from django.views.generic import TemplateView, DetailView

from core.models import ContentPost

User = get_user_model()


class HomeView(TemplateView):
    template_name = "core/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["content"] = ContentPost.objects.all()
        context["users"] = User.objects.all()
        return context


class ProfileDetailView(DetailView):
    model = User
    template_name = "core/profile-detail.html"
    slug_field = "username"
    slug_url_kwarg = "username"


class ContentPostDetailView(DetailView):
    model = ContentPost
    template_name = "core/content-post-detail.html"
    slug_field = "uuid"
    slug_url_kwarg = "uuid"
    context_object_name = "post"
