from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib.auth.views import LogoutView
from django.contrib.auth import logout as auth_logout
from django_htmx.http import HttpResponseClientRedirect


class HtmxLogoutView(LoginRequiredMixin, LogoutView):
    def post(self, request, *args, **kwargs):
        if request.htmx:
            redirect_to = self.get_success_url()
            if self.request.user.is_authenticated:
                auth_logout(request)
            return HttpResponseClientRedirect(redirect_to)

        return super().post(request, *args, **kwargs)
