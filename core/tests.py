import io
import shutil
from unittest.mock import patch

from PIL import Image
from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.management import call_command
from django.test import Client
from django.test import TestCase, override_settings
from django.urls import reverse

from core.models import ContentPost

User = get_user_model()


class BaseTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        call_command("collectstatic", "--noinput")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(settings.STATIC_ROOT)


class HomeViewTests(BaseTestCase):
    def setUp(self) -> None:
        # Create two users: one for the active user and one for a "similar" content owner.
        self.active_user = User.objects.create_user(
            email="active@example.com", username="active", password="password"
        )
        self.other_user = User.objects.create_user(
            email="other@example.com", username="other", password="password"
        )
        # Create a content post for active user (with dummy embedding) and one for other user.
        self.active_post = ContentPost.objects.create(
            user=self.active_user,
            title="Active Post",
            embedding=[0.1, 0.2, 0.3],
        )
        self.other_post = ContentPost.objects.create(
            user=self.other_user,
            title="Other User Post",
            embedding=[0.4, 0.5, 0.6],
        )

        # For testing suggested content, we monkeypatch active_post.get_similar_posts.
        # When called, it will return a list containing the other user's post.
        self.active_post.get_similar_posts = lambda **kwargs: [self.other_post]

        self.client = Client()

    def test_home_view_anonymous(self) -> None:
        """
        Anonymous users should receive fallback context with all content
        and most active users.
        """
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        # Fallback context keys.
        self.assertIn("content", response.context)
        self.assertIn("users", response.context)
        # Suggested content should not be in context.
        self.assertNotIn("suggested_content", response.context)

    def test_home_view_authenticated_no_suggestions(self) -> None:
        """
        Authenticated user with no similar posts (i.e. get_similar_posts returns empty)
        should fall back to general content.
        """
        self.client.login(email="active@example.com", password="password")
        # Monkeypatch get_similar_posts to return empty list.
        self.active_post.get_similar_posts = lambda **kwargs: []
        response = self.client.get(reverse("home"))
        self.assertEqual(response.status_code, 200)
        # Fallback context is used.
        self.assertIn("content", response.context)
        self.assertIn("users", response.context)
        self.assertNotIn("suggested_content", response.context)


class ProfileDetailViewTests(BaseTestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(
            email="profile@example.com", username="profileuser", password="password"
        )
        self.client = Client()

    def test_profile_detail_view(self) -> None:
        """
        The profile detail view should return the correct user in context.
        """
        url = reverse("profile-detail", kwargs={"username": self.user.username})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context.get("profile"), self.user)


class ContentPostDetailViewTests(BaseTestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(
            email="postdetail@example.com", username="postuser", password="password"
        )

        self.post = ContentPost.objects.create(
            user=self.user, title="Detail Post", embedding=[0.1, 0.2, 0.3]
        )
        self.client = Client()

    def test_content_post_detail_view(self) -> None:
        """
        The content post detail view should return the correct post in context.
        """
        url = reverse("content-post-detail", kwargs={"uuid": self.post.uuid})
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context.get("post"), self.post)


class CreateContentPostViewTests(BaseTestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(
            email="create@example.com", username="createuser", password="password"
        )
        self.client = Client()
        self.url = reverse("create-content-post")

    def test_create_content_post_view_get(self) -> None:
        """
        GET request to the create view should display the form.
        """
        self.client.login(email="create@example.com", password="password")
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, 200)
        # Check that the form is present in the context.
        self.assertIn("form", response.context)

    @override_settings(MEDIA_ROOT="/tmp/django_test_media")
    def test_create_content_post_view_post(self) -> None:
        """
        POST request with valid data should create a ContentPost and redirect to its detail page.
        """
        self.client.login(email="create@example.com", password="password")
        data = {
            "title": "New Post",
            "description": "Test description",
            "media_type": "text",
            "text_content": "This is a test post.",
            # Assuming embedding can be blank; in production it's computed later.
        }
        response = self.client.post(self.url, data)
        # Expect a redirect on successful creation.
        self.assertEqual(response.status_code, 302)
        new_post = ContentPost.objects.get(title="New Post")
        self.assertEqual(new_post.text_content, "This is a test post.")
        # Check that the redirect URL is the post's detail page.
        expected_redirect = new_post.get_absolute_url()
        self.assertIn(expected_redirect, response.url)


class GenerateEmbeddingsCommandTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            email="test@example.com", username="testuser", password="password"
        )

        self.text_post = ContentPost.objects.create(
            user=self.user,
            title="Text Post",
            media_type="text",
            text_content="This is a sample text for embedding.",
        )

        # Create an image post with a dummy image

        image_io = io.BytesIO()
        image = Image.new("RGB", (100, 100), color="red")
        image.save(image_io, format="JPEG")
        image_io.seek(0)
        image_file = SimpleUploadedFile(
            "test.jpg", image_io.read(), content_type="image/jpeg"
        )
        self.image_post = ContentPost.objects.create(
            user=self.user,
            title="Image Post",
            media_type="image",
            media_file=image_file,
        )

        # Create an audio post with dummy content (not a valid audio file, but we'll patch the processing)
        audio_file = SimpleUploadedFile(
            "test.mp3", b"Fake audio data", content_type="audio/mpeg"
        )
        self.audio_post = ContentPost.objects.create(
            user=self.user,
            title="Audio Post",
            media_type="audio",
            media_file=audio_file,
        )

        # Create a video post with dummy content
        video_file = SimpleUploadedFile(
            "test.mp4", b"Fake video data", content_type="video/mp4"
        )
        self.video_post = ContentPost.objects.create(
            user=self.user,
            title="Video Post",
            media_type="video",
            media_file=video_file,
        )

    @patch(
        "core.management.commands.genembeddings.EmbeddingProcessor.compute_text_embedding"
    )
    def test_text_post_embedding(self, mock_compute_text_embedding):
        # Simulate a known text embedding
        mock_compute_text_embedding.return_value = [0.1, 0.2, 0.3]
        call_command("genembeddings", content_id=self.text_post.id)
        self.text_post.refresh_from_db()
        self.assertEqual(self.text_post.embedding, [0.1, 0.2, 0.3])

    @patch(
        "core.management.commands.genembeddings.EmbeddingProcessor.compute_image_embedding"
    )
    def test_image_post_embedding(self, mock_compute_image_embedding):
        # Simulate a known image embedding
        mock_compute_image_embedding.return_value = [0.4, 0.5, 0.6]
        call_command("genembeddings", content_id=self.image_post.id)
        self.image_post.refresh_from_db()
        self.assertEqual(self.image_post.embedding, [0.4, 0.5, 0.6])

    @patch("core.management.commands.genembeddings.EmbeddingProcessor.transcribe_audio")
    @patch(
        "core.management.commands.genembeddings.EmbeddingProcessor.compute_text_embedding"
    )
    @patch(
        "core.management.commands.genembeddings.EmbeddingProcessor.compute_audio_embedding"
    )
    def test_audio_post_embedding_with_transcription(
        self,
        mock_compute_audio_embedding,
        mock_compute_text_embedding,
        mock_transcribe_audio,
    ):
        # Simulate successful transcription for audio post
        mock_transcribe_audio.return_value = "Transcribed audio text"
        mock_compute_text_embedding.return_value = [0.7, 0.8, 0.9]
        call_command("genembeddings", content_id=self.audio_post.id)
        self.audio_post.refresh_from_db()
        self.assertEqual(self.audio_post.embedding, [0.7, 0.8, 0.9])

    @patch("core.management.commands.genembeddings.EmbeddingProcessor.transcribe_audio")
    @patch(
        "core.management.commands.genembeddings.EmbeddingProcessor.compute_audio_embedding"
    )
    def test_audio_post_embedding_without_transcription(
        self, mock_compute_audio_embedding, mock_transcribe_audio
    ):
        # Simulate failed transcription (empty transcript) for audio post
        mock_transcribe_audio.return_value = ""
        mock_compute_audio_embedding.return_value = [1.0, 1.1, 1.2]
        call_command("genembeddings", content_id=self.audio_post.id)
        self.audio_post.refresh_from_db()
        self.assertEqual(self.audio_post.embedding, [1.0, 1.1, 1.2])

    @patch("core.management.commands.genembeddings.subprocess.run")
    @patch("core.management.commands.genembeddings.EmbeddingProcessor.transcribe_audio")
    @patch(
        "core.management.commands.genembeddings.EmbeddingProcessor.compute_text_embedding"
    )
    def test_video_post_embedding_with_transcription(
        self, mock_compute_text_embedding, mock_transcribe_audio, mock_subprocess_run
    ):
        # Simulate FFmpeg extraction success and successful transcription for video post
        mock_subprocess_run.return_value.returncode = 0
        mock_transcribe_audio.return_value = "Video transcript text"
        mock_compute_text_embedding.return_value = [1.3, 1.4, 1.5]
        call_command("genembeddings", content_id=self.video_post.id)
        self.video_post.refresh_from_db()
        self.assertEqual(self.video_post.embedding, [1.3, 1.4, 1.5])

    @patch("core.management.commands.genembeddings.subprocess.run")
    @patch("core.management.commands.genembeddings.EmbeddingProcessor.transcribe_audio")
    @patch(
        "core.management.commands.genembeddings.EmbeddingProcessor.compute_audio_embedding"
    )
    def test_video_post_embedding_without_transcription(
        self, mock_compute_audio_embedding, mock_transcribe_audio, mock_subprocess_run
    ):
        # Simulate FFmpeg extraction success but transcription fails (empty), so fallback to audio embedding
        mock_subprocess_run.return_value.returncode = 0
        mock_transcribe_audio.return_value = ""
        mock_compute_audio_embedding.return_value = [1.6, 1.7, 1.8]
        call_command("genembeddings", content_id=self.video_post.id)
        self.video_post.refresh_from_db()
        self.assertEqual(self.video_post.embedding, [1.6, 1.7, 1.8])
