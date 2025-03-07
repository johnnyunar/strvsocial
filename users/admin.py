from django.contrib import admin
from django.contrib.auth.admin import GroupAdmin as BaseGroupAdmin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import Group
from django.utils.translation import gettext_lazy as _
from unfold.admin import ModelAdmin

from users.models import SocialUser

admin.site.unregister(Group)


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


@admin.register(Group)
class GroupAdmin(BaseGroupAdmin, ModelAdmin):
    pass
