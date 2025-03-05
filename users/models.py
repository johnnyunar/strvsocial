from functools import partial

from django.contrib.auth.models import AbstractUser
from django.db import models
from django.templatetags.static import static

from core.utils import generate_random_filename
from users.manager import SocialUserManager


class SocialUser(AbstractUser):
    email = models.EmailField(unique=True)
    bio = models.TextField(blank=True, max_length=255)
    avatar = models.ImageField(
        upload_to=partial(generate_random_filename, subdir="avatars"),
        blank=True,
        null=True,
    )
    objects = SocialUserManager()

    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = ["username"]

    def __str__(self):
        return self.email

    @property
    def user(self):
        """Return the user instance. This is a convenience method for consistency."""
        return self

    def get_last_update(self):
        content = self.content.latest()
        return content.updated_at

    def get_avatar_url(self):
        if self.avatar:
            return self.avatar.url

        return static("core/img/user.webp")
