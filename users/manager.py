from datetime import timedelta

from django.contrib.auth.models import UserManager
from django.db.models import Q
from django.db.models.aggregates import Count
from django.utils import timezone


class SocialUserManager(UserManager):
    def get_most_active(self, limit: int=5, time_delta: timedelta=timedelta(days=30)):
        """
        Return N most active users by post count in the given time delta.

        :param limit: Number of users to return
        :param time_delta: Time delta to consider for post count
        :return: QuerySet of filtered users
        """
        return self.annotate(
            post_count=Count(
                "content",
                filter=Q(content__created_at__gte=timezone.now() - time_delta),
            )
        ).order_by("-post_count")[:limit]