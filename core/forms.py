from django import forms
from .models import ContentPost

from django.utils.translation import gettext_lazy as _


class ContentPostForm(forms.ModelForm):
    class Meta:
        model = ContentPost
        fields = ["title", "description", "media_file", "text_content"]
        widgets = {
            "title": forms.TextInput(attrs={"placeholder": _("My Awesome Post")}),
            "description": forms.Textarea(
                attrs={
                    "placeholder": _("A detailed description of the content..."),
                }
            ),
            "media_file": forms.ClearableFileInput(),
            "text_content": forms.Textarea(
                attrs={
                    "placeholder": "Write something...",
                }
            ),
        }
