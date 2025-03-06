import numpy as np
import faiss
from typing import List, Tuple, Dict

from django.contrib.auth import get_user_model
from django.core.cache import cache

from core.models import ContentPost

User = get_user_model()

CACHE_KEY_PREFIX = "faiss_indexes__"


def get_faiss_indexes_from_cache(
    cache_key: str = CACHE_KEY_PREFIX,
    media_types=None,
) -> Dict[str, Tuple[faiss.IndexFlatL2, List[int]]] | None:
    """
    Get FAISS indexes from cache.

    :param cache_key: The key to use for the cache lookup.
    :return: A dictionary mapping media types to a tuple containing:
        - A FAISS IndexFlatL2 object built from embeddings of that media type.
        - A list mapping each index position to a ContentPost ID.
    """
    if media_types is None:
        media_types = ["text", "image", "gif", "audio"]

    all_indexes = {}
    for media_type in media_types:
        cached = cache.get(f"{cache_key}{media_type}")
        if cached:
            # Rebuild the indexes from the serialized data in cache.
            indexes: Dict[str, Tuple[faiss.IndexFlatL2, List[int]]] = {}
            for media_type, (serialized_index, id_list) in cached.items():
                index = faiss.deserialize_index(serialized_index)
                all_indexes[media_type] = (index, id_list)

    return all_indexes or None


def build_faiss_indexes_by_media(
    force_rebuild: bool = False,
    media_types=None,
) -> Dict[str, Tuple[faiss.IndexFlatL2, List[int]]]:
    """
    Build and cache separate FAISS indexes for each media type.

    This function builds FAISS indexes for each media type (text, image, gif, audio)
    based on ContentPost embeddings. The indexes are cached in Redis using Django's
    caching framework. When the cache is hit, it deserializes the stored FAISS index
    bytes back into a FAISS index.

    Args:
        force_rebuild: If True, rebuild the indexes even if they are cached.
        media_types: List of media types to build indexes for. Default is ["text", "image", "gif", "audio"].

    Returns:
        A dictionary mapping each media type to a tuple containing:
          - A FAISS IndexFlatL2 object built from embeddings of that media type.
          - A list mapping each index position to a ContentPost ID.
    """
    if media_types is None:
        media_types = ["text", "image", "gif", "audio"]

    if not force_rebuild:
        if cached_indexes := get_faiss_indexes_from_cache(media_types):
            return cached_indexes

    # Build the indexes if not found in cache.
    indexes: Dict[str, Tuple[faiss.IndexFlatL2, List[int]]] = {}

    for media_type in media_types:
        qs = ContentPost.objects.filter(embedding__isnull=False, media_type=media_type)
        embeddings_list: List[List[float]] = []
        id_list: List[int] = []

        for post in qs:
            embeddings_list.append(post.embedding)
            id_list.append(post.id)

        if embeddings_list:
            embeddings_np = np.array(embeddings_list).astype("float32")
            _, dim = embeddings_np.shape
            index = faiss.IndexFlatL2(dim)
            index.add(embeddings_np)
            indexes[media_type] = (index, id_list)
            # Serialize the FAISS index to bytes for caching.
            serialized_index = faiss.serialize_index(index)
            # Cache the serialized index and ID list for an hour.
            cache.set(
                f"{CACHE_KEY_PREFIX}{media_type}",
                (serialized_index, id_list),
                timeout=3600,
            )

    return indexes


def get_similar_for_post(
    query_embedding: List[float],
    index: faiss.IndexFlatL2,
    id_list: List[int],
    query_user_id: int,
    k: int = 5,
    search_multiplier: int = 2,
    threshold: float | None = None,
) -> List[ContentPost]:
    """
    Find similar ContentPost instances for a given query embedding,
    filtering out posts from the same user and optionally applying a distance threshold.

    Args:
        query_embedding: The embedding vector of the query post.
        index: The FAISS index corresponding to the media type.
        id_list: List mapping FAISS index positions to ContentPost IDs.
        query_user_id: ID of the user who owns the query post.
        k: Number of similar posts to return.
        search_multiplier: Multiplier to fetch extra candidates for filtering.
        threshold: Optional distance threshold. Only candidates with a distance
                   less than or equal to this value will be returned.

    Returns:
        A list of ContentPost instances from other users that are similar,
        possibly fewer than k if candidates don't meet the threshold.
    """
    search_k = k * search_multiplier
    query_vec = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vec, search_k)
    similar_posts: List[ContentPost] = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx >= len(id_list):
            continue

        candidate_id = id_list[idx]
        candidate_post = ContentPost.objects.get(id=candidate_id)

        # Skip posts from the same user.
        if candidate_post.user.id == query_user_id:
            continue

        # If a threshold is set, stop processing if the candidate is too far.
        if threshold is not None and dist > threshold:
            break

        similar_posts.append(candidate_post)
        if len(similar_posts) == k:
            break

    return similar_posts
