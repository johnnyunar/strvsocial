from django.contrib import admin
from unfold.admin import ModelAdmin

from core.models import ContentPost


@admin.register(ContentPost)
class ContentPostAdmin(ModelAdmin):
    list_display = ("title", "media_type", "user", "created_at", "updated_at")
    search_fields = ("title",)
    date_hierarchy = "created_at"
    list_filter = ("created_at", "updated_at")
    ordering = ("-created_at",)
