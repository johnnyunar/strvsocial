import shutil
import tempfile
from contextlib import contextmanager
from datetime import timedelta

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.management import call_command
from django.templatetags.static import static
from django.test import Client
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from django_htmx.http import HttpResponseClientRedirect

from core.models import ContentPost

User = get_user_model()

class SocialUserModelTests(TestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(
            email="test@example.com", username="testuser", password="password"
        )

    @classmethod
    def setUpClass(cls) -> None:
        call_command("collectstatic", "--noinput")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(settings.STATIC_ROOT)

    def test_str_returns_email(self) -> None:
        """Ensure __str__ returns the user's email."""
        self.assertEqual(str(self.user), self.user.email)

    def test_get_avatar_url_without_avatar(self) -> None:
        """When no avatar is set, get_avatar_url should return the default static URL."""
        expected = static("core/img/user.webp")
        self.assertEqual(self.user.get_avatar_url(), expected)

    def test_get_avatar_url_with_avatar(self) -> None:
        """When an avatar is set, get_avatar_url should return its URL."""
        fake_avatar = type("FakeFieldFile", (), {"url": "http://example.com/avatar.png"})()
        self.user.avatar = fake_avatar
        self.assertEqual(self.user.get_avatar_url(), "http://example.com/avatar.png")

    def test_get_last_update_returns_latest_post_update(self) -> None:
        """
        Ensure get_last_update returns the updated_at of the latest ContentPost.
        """
        now = timezone.now()
        post1 = ContentPost.objects.create(
            user=self.user, title="Post 1", embedding=[0.1, 0.2]
        )
        post1.updated_at = now
        post1.save()
        post2 = ContentPost.objects.create(
            user=self.user, title="Post 2", embedding=[0.3, 0.4]
        )
        later = now + timedelta(hours=1)
        post2.updated_at = later
        post2.save()

        self.assertEqual(self.user.get_last_update(), post2.updated_at)


class SocialUserManagerTests(TestCase):
    def setUp(self) -> None:
        # Create three users.
        self.user1 = User.objects.create_user(
            email="user1@example.com", username="user1", password="pass"
        )
        self.user2 = User.objects.create_user(
            email="user2@example.com", username="user2", password="pass"
        )
        self.user3 = User.objects.create_user(
            email="user3@example.com", username="user3", password="pass"
        )

        now = timezone.now()
        # User1 has 3 posts within the last 30 days.
        for title in ["Post A", "Post B", "Post C"]:
            post = ContentPost.objects.create(
                user=self.user1, title=title, embedding=[0.1, 0.2]
            )
            post.created_at = now - timedelta(days=5)
            post.save()

        # User2 has 1 post within the last 30 days.
        post = ContentPost.objects.create(
            user=self.user2, title="Post D", embedding=[0.3, 0.4]
        )
        post.created_at = now - timedelta(days=3)
        post.save()

        # User3 has a post older than 30 days.
        post = ContentPost.objects.create(
            user=self.user3, title="Old Post", embedding=[0.5, 0.6]
        )
        post.created_at = now - timedelta(days=40)
        post.save()

    def test_get_most_active_returns_correct_users(self) -> None:
        """
        get_most_active should return user1 (3 posts) and user2 (1 post),
        excluding user3 with no recent posts.
        """
        active_users = list(User.objects.get_most_active(limit=2, time_delta=timedelta(days=30)))
        self.assertEqual(active_users, [self.user1, self.user2])

    def test_get_most_active_returns_correct_users_plus_one(self) -> None:
        """
        get_most_active should return user1 (3 posts) and user2 (1 post),
        excluding user3 with no recent posts.
        """
        active_users = list(User.objects.get_most_active(limit=3, time_delta=timedelta(days=30)))
        self.assertEqual(active_users, [self.user1, self.user2, self.user3])

    def test_get_most_active_limit(self) -> None:
        """
        Ensure get_most_active honors the limit parameter.
        """
        active_users = list(User.objects.get_most_active(limit=1, time_delta=timedelta(days=30)))
        self.assertEqual(len(active_users), 1)
        self.assertEqual(active_users[0], self.user1)


class HtmxLogoutViewTests(TestCase):
    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user(
            email="test2@example.com", username="testuser2", password="password"
        )
        # Adjust the URL name as per your URL configuration.
        self.logout_url = reverse("logout")

    def test_htmx_logout(self) -> None:
        """
        An HTMX POST request to the logout view should log out the user and return
        an HttpResponseClientRedirect.
        """
        self.client.login(email="test2@example.com", password="password")
        response = self.client.post(self.logout_url, HTTP_HX_REQUEST="true")
        self.assertIsInstance(response, HttpResponseClientRedirect)
        self.assertNotIn("_auth_user_id", self.client.session)

    def test_regular_logout(self) -> None:
        """
        A standard POST request to logout should log out the user and return a 302 redirect.
        """
        self.client.login(email="test2@example.com", password="password")
        response = self.client.post(self.logout_url)
        self.assertEqual(response.status_code, 302)
        self.assertNotIn("_auth_user_id", self.client.session)
