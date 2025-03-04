import uuid
from django.db import models

from django.utils.translation import gettext_lazy as _


class BaseModel(models.Model):
    uuid = models.UUIDField(_("UUID"), default=uuid.uuid4, editable=False, unique=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        abstract = True
