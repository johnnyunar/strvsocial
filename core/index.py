import numpy as np
import faiss
from typing import List, Tuple, Dict

from django.contrib.auth import get_user_model
from django.core.cache import cache

from core.models import ContentPost

User = get_user_model()

CACHE_KEY = "faiss_indexes_by_media"


def get_faiss_indexes_from_cache(
    cache_key: str = CACHE_KEY,
) -> Dict[str, Tuple[faiss.IndexFlatL2, List[int]]] | None:
    cached = cache.get(cache_key)
    if cached:
        # Rebuild the indexes from the serialized data in cache.
        indexes: Dict[str, Tuple[faiss.IndexFlatL2, List[int]]] = {}
        for media_type, (serialized_index, id_list) in cached.items():
            index = faiss.deserialize_index(serialized_index)
            indexes[media_type] = (index, id_list)
        return indexes


def build_faiss_indexes_by_media(
    force_rebuild: bool = False,
) -> Dict[str, Tuple[faiss.IndexFlatL2, List[int]]]:
    """
    Build and cache separate FAISS indexes for each media type.

    This function builds FAISS indexes for each media type (text, image, gif, audio)
    based on ContentPost embeddings. The indexes are cached in Redis using Django's
    caching framework. When the cache is hit, it deserializes the stored FAISS index
    bytes back into a FAISS index.

    Args:
        force_rebuild: If True, rebuild the indexes even if they are cached.

    Returns:
        A dictionary mapping each media type to a tuple containing:
          - A FAISS IndexFlatL2 object built from embeddings of that media type.
          - A list mapping each index position to a ContentPost ID.
    """
    if not force_rebuild:
        if cached_indexes := get_faiss_indexes_from_cache():
            return cached_indexes

    # Build the indexes if not found in cache.
    indexes: Dict[str, Tuple[faiss.IndexFlatL2, List[int]]] = {}
    result_to_cache: Dict[str, Tuple[bytes, List[int]]] = {}
    media_types = ["text", "image", "gif", "audio"]

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
            result_to_cache[media_type] = (serialized_index, id_list)

    # Cache the serialized indexes for a defined timeout (1 hour).
    cache.set(CACHE_KEY, result_to_cache, timeout=3600)
    return indexes


def get_similar_for_post(
    query_embedding: List[float],
    index: faiss.IndexFlatL2,
    id_list: List[int],
    query_user_id: int,
    k: int = 5,
    search_multiplier: int = 2,
) -> List[ContentPost]:
    """
    Find similar ContentPost IDs for a given query embedding,
    filtering out posts from the same user.

    Args:
        query_embedding: The embedding vector of the query post.
        index: The FAISS index corresponding to the media type.
        id_list: List mapping FAISS index positions to ContentPost IDs.
        query_user_id: ID of the user who owns the query post.
        k: Number of similar posts to return.
        search_multiplier: Multiplier to fetch extra candidates for filtering.

    Returns:
        A list of ContentPost IDs from other users that are similar.
    """
    search_k = k * search_multiplier
    query_vec = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vec, search_k)
    similar_ids: List[ContentPost] = []

    for idx in indices[0]:
        if idx < len(id_list):
            candidate_id = id_list[idx]
            candidate_post = ContentPost.objects.get(id=candidate_id)
            if candidate_post.user_id != query_user_id:
                similar_ids.append(candidate_post)
            if len(similar_ids) == k:
                break
    return similar_ids
