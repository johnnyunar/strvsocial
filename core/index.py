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
        media_types = ["text", "image", "video", "gif", "audio"]

    if not force_rebuild:
        if existing_indexes := get_faiss_indexes_from_cache(media_types):
            return existing_indexes

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
    Retrieve similar ContentPost instances for a given query embedding,
    filtering out posts from the same user and applying an optional threshold.

    This implementation leverages NumPy for vectorized filtering and performs
    a single batch database query to retrieve candidate posts.

    Args:
        query_embedding: The embedding vector of the query post.
        index: The FAISS index for the corresponding media type.
        id_list: Mapping from FAISS index positions to ContentPost IDs.
        query_user_id: The user ID of the querying post.
        k: Maximum number of similar posts to return.
        search_multiplier: Overfetch factor to account for filtering.
        threshold: Optional maximum L2 distance; only candidates with a distance
                   <= threshold are considered.

    Returns:
        A list of ContentPost instances from other users that are similar,
        possibly fewer than k if candidates don’t meet the threshold.
    """
    # Overfetch to allow for filtering
    search_k = k * search_multiplier
    query_vec = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vec, search_k)

    # Create a boolean mask for candidates under the threshold.
    # If threshold is None, accept all candidates.
    mask = (
        np.ones_like(distances[0], dtype=bool)
        if threshold is None
        else (distances[0] <= threshold)
    )

    # Get valid indices according to the mask.
    valid_positions = np.where(mask)[0]
    if valid_positions.size == 0:
        return []

    # Retrieve candidate IDs in the order they were returned.
    candidate_ids = [
        id_list[int(idx)] for idx in indices[0][valid_positions] if idx < len(id_list)
    ]

    # Exclude candidates from the same user in one batch query.
    candidate_posts = list(
        ContentPost.objects.filter(id__in=candidate_ids).exclude(user__id=query_user_id)
    )

    # Create a mapping from candidate_id to distance for sorting.
    distance_map = {}
    for pos in valid_positions:
        idx = int(indices[0][pos])
        if idx < len(id_list):
            candidate_id = id_list[idx]
            distance_map[candidate_id] = float(distances[0][pos])

    # Sort candidates by distance.
    candidate_posts.sort(key=lambda post: distance_map.get(post.id, float("inf")))

    return candidate_posts[:k]
