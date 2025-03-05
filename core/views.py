import logging
from typing import Dict, List, Tuple, Any

from django.contrib.auth import get_user_model
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView, DetailView, CreateView

from core.forms import ContentPostForm
from core.index import (
    get_similar_for_post,
    build_faiss_indexes_by_media,
)
from core.models import ContentPost

logger = logging.getLogger(__name__)

User = get_user_model()


class HomeView(TemplateView):
    template_name = "core/home.html"

    def get_context_data(self, **kwargs) -> dict:
        """
        Build the context with all users and suggested content.
        Suggested content is built by grouping similar posts from other users
        and, for each of them, listing the active user's posts that are similar.
        """
        context: dict = super().get_context_data(**kwargs)
        context["users"] = User.objects.all()

        # This dict maps each active user's post ID to the list of similar posts (from other users)
        active_to_similar: Dict[int, List[ContentPost]] = {}

        if self.request.user.is_authenticated:
            # Build separate FAISS indexes for each media type
            faiss_indexes = build_faiss_indexes_by_media()

            # Get active user's posts that have embeddings
            active_user_posts = ContentPost.objects.filter(
                user=self.request.user, embedding__isnull=False
            )

            for active_post in active_user_posts:
                media_type = active_post.media_type
                # Process only if there's an index for this media type
                if media_type in faiss_indexes:
                    index, id_list = faiss_indexes[media_type]
                    similar_posts = get_similar_for_post(
                        query_embedding=active_post.embedding,
                        index=index,
                        id_list=id_list,
                        query_user_id=self.request.user.id,
                        k=5,
                    )
                    if similar_posts:
                        active_to_similar[active_post.id] = similar_posts

            # Invert the mapping:
            # For each similar post (from another user), collect the active user's posts that are similar to it.
            suggested_content: Dict[int, Dict[str, Any]] = {}
            for active_post_id, similar_posts in active_to_similar.items():
                # Get the active post instance
                try:
                    active_post = ContentPost.objects.get(id=active_post_id)
                except ContentPost.DoesNotExist:
                    continue
                for other_post in similar_posts:
                    # Use the other user's post ID as the key.
                    if other_post.id in suggested_content:
                        suggested_content[other_post.id]["active_posts"].append(
                            active_post
                        )
                    else:
                        suggested_content[other_post.id] = {
                            "post": other_post,
                            "active_posts": [active_post],
                        }

            if suggested_content:
                context["suggested_content"] = suggested_content
                similar_users = User.objects.filter(
                    content__in=suggested_content.keys()
                ).distinct()
                context["users"] = similar_users
            else:
                context["content"] = ContentPost.objects.all()

        return context


class ProfileDetailView(DetailView):
    model = User
    template_name = "core/profile-detail.html"
    slug_field = "username"
    slug_url_kwarg = "username"


class ContentPostDetailView(DetailView):
    model = ContentPost
    template_name = "core/content-post-detail.html"
    slug_field = "uuid"
    slug_url_kwarg = "uuid"
    context_object_name = "post"


class CreateContentPostView(LoginRequiredMixin, CreateView):
    template_name = "core/create-content-post.html"
    form_class = ContentPostForm

    def get_success_url(self):
        return self.object.get_absolute_url()
