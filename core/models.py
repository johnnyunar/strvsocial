import uuid
from functools import partial

import magic
from django.contrib.auth import get_user_model
from django.db import models
from django.urls import reverse
from django.utils.translation import gettext_lazy as _
from django_currentuser.db.models import CurrentUserField

from core.const import SIMILARITY_THRESHOLDS
from core.utils import generate_random_filename
from core.validators import validate_media_file

User = get_user_model()


class BaseModel(models.Model):
    uuid = models.UUIDField(_("UUID"), default=uuid.uuid4, editable=False, unique=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    is_active = models.BooleanField(default=True)

    class Meta:
        abstract = True


class ContentPost(BaseModel):
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

    user = CurrentUserField(
        related_name="content", on_delete=models.CASCADE, verbose_name=_("User")
    )
    title = models.CharField(_("Title"), max_length=255)
    description = models.TextField(_("Description"), blank=True, null=True)
    text_content = models.TextField(_("Text Content"), blank=True, null=True)
    media_type = models.CharField(
        _("Media Type"), max_length=10, choices=MEDIA_TYPE_CHOICES
    )
    media_file = models.FileField(
        _("Media File"),
        upload_to=partial(generate_random_filename, subdir="content"),
        blank=True,
        null=True,
        validators=[validate_media_file],
    )
    embedding = models.JSONField(_("Embedding"), blank=True, null=True)

    class Meta:
        verbose_name = _("Content")
        verbose_name_plural = _("Content")
        ordering = ["-created_at"]
        get_latest_by = "updated_at"

    def save(self, *args, **kwargs):
        if self.media_file and not self.media_type:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_buffer(self.media_file.read(2048))

            if mime_type.startswith("image"):
                self.media_type = "image" if "gif" not in mime_type else "gif"
            elif mime_type.startswith("video"):
                self.media_type = "video"
            elif mime_type.startswith("audio"):
                self.media_type = "audio"
        elif self.text_content and not self.media_type:
            self.media_type = "text"

        super().save(*args, **kwargs)

    def get_absolute_url(self):
        return reverse("content-post-detail", kwargs={"uuid": self.uuid})

    def __str__(self) -> str:
        return f"{self.title} ({self.media_type})"

    def get_similar_posts(
        self,
        index: "faiss.Index" = None,
        id_list: list[int] = None,
        query_user_id: int = None,
        k: int = 5,
    ) -> list["ContentPost"]:
        from core.index import get_similar_for_post, build_faiss_indexes_by_media

        if not index or not id_list:
            faiss_indexes = build_faiss_indexes_by_media(
                force_rebuild=True, media_types=[self.media_type]
            )
            index, id_list = faiss_indexes.get(self.media_type)
            if not index:
                return []

        if not query_user_id:
            query_user_id = self.user.id

        return get_similar_for_post(
            query_embedding=self.embedding,
            index=index,
            id_list=id_list,
            query_user_id=query_user_id,
            k=k,
            threshold=SIMILARITY_THRESHOLDS.get(self.media_type, 500.0),
        )
