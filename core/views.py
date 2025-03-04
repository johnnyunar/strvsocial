from django.contrib.auth import get_user_model
from django.views.generic import TemplateView

from core.models import Content

User = get_user_model()


class HomeView(TemplateView):
    template_name = "core/home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["content"] = Content.objects.all()
        context["users"] = User.objects.all()
        return context
