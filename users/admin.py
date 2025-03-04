from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from unfold.admin import ModelAdmin

from users.models import SocialUser

from django.utils.translation import gettext_lazy as _


@admin.register(SocialUser)
class SocialUserAdmin(UserAdmin, ModelAdmin):
    model = SocialUser
    list_display = ("email", "username", "first_name", "last_name", "is_staff")
    add_fieldsets = UserAdmin.add_fieldsets + ((None, {"fields": ("email",)}),)
    search_fields = ("email", "username")

    fieldsets = (
        (None, {"fields": ("username", "password")}),
        (
            _("Personal info"),
            {"fields": ("first_name", "last_name", "email", "bio", "avatar")},
        ),
        (
            _("Permissions"),
            {
                "fields": (
                    "is_active",
                    "is_staff",
                    "is_superuser",
                    "groups",
                    "user_permissions",
                ),
            },
        ),
        (_("Important dates"), {"fields": ("last_login", "date_joined")}),
    )
