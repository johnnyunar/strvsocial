import uuid

from django.contrib.auth import get_user_model
from django.db import models
from django.utils.translation import gettext_lazy as _

User = get_user_model()


class BaseModel(models.Model):
    uuid = models.UUIDField(_("UUID"), default=uuid.uuid4, editable=False, unique=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        abstract = True


class Content(BaseModel):
    """
    Model representing a content item in the social network.

    Attributes:
        user: The uploader/owner of the content.
        title: A short title for the content.
        description: A detailed description of the content.
        media_type: The type of media (image, gif, text, audio, video).
        media_file: The file field for images, gifs, audio, or video content.
        text_content: Textual content for posts.
        embedding: A JSON field to store computed embeddings.
    """

    MEDIA_TYPE_CHOICES = [
        ("image", "Image"),
        ("gif", "GIF"),
        ("text", "Text"),
        ("audio", "Audio"),
        ("video", "Video"),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="content")
    title = models.CharField(_("Title"), max_length=255)
    description = models.TextField(_("Description"), blank=True, null=True)
    text_content = models.TextField(_("Text Content"), blank=True, null=True)
    media_type = models.CharField(
        _("Media Type"), max_length=10, choices=MEDIA_TYPE_CHOICES
    )
    media_file = models.FileField(
        _("Media File"), upload_to="uploads/", blank=True, null=True
    )
    embedding = models.JSONField(_("Embedding"), blank=True, null=True)

    def __str__(self) -> str:
        return f"{self.title} ({self.media_type})"
