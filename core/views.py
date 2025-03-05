import logging
from datetime import timedelta
from typing import Dict, List, Any

from django.contrib.auth import get_user_model
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, DetailView, CreateView

from core.forms import ContentPostForm
from core.index import build_faiss_indexes_by_media
from core.models import ContentPost

logger = logging.getLogger(__name__)
User = get_user_model()


class HomeView(TemplateView):
    template_name = "core/home.html"

    def get_context_data(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Build the context for the home page.

        If the user is authenticated, compute suggested content by grouping similar posts
        from other users and associating them with the active user's posts. Otherwise,
        fallback to general content and a list of most active users.
        """
        context: Dict[str, Any] = super().get_context_data(**kwargs)
        if self.request.user.is_authenticated:
            suggested = self._build_suggested_content()
            if suggested:
                context["suggested_content"] = suggested
                context["users"] = self._get_similar_users(suggested)
                return context

        context["content"] = ContentPost.objects.all()
        context["users"] = User.objects.get_most_active(limit=5, time_delta=timedelta(days=30))
        return context

    def _build_suggested_content(self) -> Dict[int, Dict[str, Any]]:
        """
        Build suggested content by inverting a mapping from active user's posts to similar posts.

        Returns:
            A dict mapping each other user's post ID to a dictionary containing:
                - "post": the other user's ContentPost instance.
                - "active_posts": a list of active user's posts similar to it.
        """
        active_to_similar = self._get_active_to_similar()
        return self._invert_active_to_similar(active_to_similar)

    def _get_active_to_similar(self) -> Dict[int, List[ContentPost]]:
        """
        Build a mapping from each active user's post ID to a list of similar posts (from other users).

        Returns:
            A dictionary where keys are active post IDs and values are lists of similar ContentPost instances.
        """
        mapping: Dict[int, List[ContentPost]] = {}
        faiss_indexes = build_faiss_indexes_by_media(force_rebuild=True)
        active_posts = ContentPost.objects.filter(
            user=self.request.user, embedding__isnull=False
        )
        for active_post in active_posts:
            media_type = active_post.media_type
            if media_type not in faiss_indexes:
                continue
            index, id_list = faiss_indexes[media_type]
            similar_posts = active_post.get_similar_posts(
                index=index,
                id_list=id_list,
                query_user_id=self.request.user.id,
                k=5,
            )
            if similar_posts:
                mapping[active_post.id] = similar_posts
        return mapping

    def _invert_active_to_similar(
        self, mapping: Dict[int, List[ContentPost]]
    ) -> Dict[int, Dict[str, Any]]:
        """
        Invert the mapping from active user's posts to similar posts.

        For each similar post (from another user), collect the active user's posts that are similar.

        Returns:
            A dictionary mapping a similar post ID to a dict with:
                - "post": the ContentPost instance (from another user).
                - "active_posts": a list of active user's ContentPost instances similar to it.
        """
        inverted: Dict[int, Dict[str, Any]] = {}
        for active_id, similar_list in mapping.items():
            try:
                active_post = ContentPost.objects.get(id=active_id)
            except ContentPost.DoesNotExist:
                continue
            for other_post in similar_list:
                if other_post.id in inverted:
                    if active_post not in inverted[other_post.id]["active_posts"]:
                        inverted[other_post.id]["active_posts"].append(active_post)
                else:
                    inverted[other_post.id] = {"post": other_post, "active_posts": [active_post]}
        return inverted

    def _get_similar_users(self, suggested: Dict[int, Dict[str, Any]]) -> List[User]:
        """
        Retrieve users who own the suggested posts.

        Args:
            suggested: The dictionary of suggested content.

        Returns:
            A list of User instances owning the suggested posts.
        """
        suggested_ids = list(suggested.keys())
        return list(User.objects.filter(content__id__in=suggested_ids).distinct())


class ProfileDetailView(DetailView):
    model = User
    template_name = "core/profile-detail.html"
    slug_field = "username"
    slug_url_kwarg = "username"
    context_object_name = "profile"


class ContentPostDetailView(DetailView):
    model = ContentPost
    template_name = "core/content-post-detail.html"
    slug_field = "uuid"
    slug_url_kwarg = "uuid"
    context_object_name = "post"


class CreateContentPostView(LoginRequiredMixin, CreateView):
    template_name = "core/create-content-post.html"
    form_class = ContentPostForm

    def get_success_url(self) -> str:
        return self.object.get_absolute_url()
