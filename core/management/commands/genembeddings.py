import logging
import os
import tempfile
from typing import List, Optional

import numpy as np
import torch
import librosa
from PIL import Image
from django.core.management.base import BaseCommand
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import AutoTokenizer, AutoModel
from core.models import ContentPost

logger = logging.getLogger(__name__)

# Global model and tokenizer references
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_tokenizer: Optional[AutoTokenizer] = None
text_model: Optional[AutoModel] = None
image_model: Optional[torch.nn.Module] = None
image_transform: Optional[transforms.Compose] = None


def init_text_model() -> None:
    """
    Initialize the text model and tokenizer if not already loaded.
    """
    global text_tokenizer, text_model
    if text_model is None:
        text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        text_model.to(device)
        text_model.eval()


def compute_text_embedding(text: str) -> List[float]:
    """
    Compute and return the text embedding for the given text.

    Args:
        text: Input text to embed.

    Returns:
        A list of floats representing the text embedding.
    """
    init_text_model()
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
    # Mean-pool over token embeddings.
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding.tolist()


def init_image_model() -> None:
    """
    Initialize the image model and its transformation pipeline if not already initialized.
    """
    global image_model, image_transform
    if image_model is None:
        base_model = models.resnet18(pretrained=True)
        # Remove the classification head.
        modules = list(base_model.children())[:-1]
        image_model = torch.nn.Sequential(*modules)
        image_model.to(device)
        image_model.eval()
        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def compute_image_embedding_from_file(file_obj) -> List[float]:
    """
    Compute an image embedding from a file-like object using a pre-trained CNN.

    Args:
        file_obj: A binary file-like object containing image data.

    Returns:
        A list of floats representing the image embedding.
    """
    init_image_model()
    image = Image.open(file_obj).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = image_model(image_tensor).squeeze().cpu().numpy()
    return embedding.flatten().tolist()


def compute_audio_embedding(audio_path: str) -> List[float]:
    """
    Compute an audio embedding using a mel-spectrogram based approach.

    This is a placeholder implementation; in production, consider using a dedicated model (e.g., VGGish).

    Args:
        audio_path: Local file path to the audio file.

    Returns:
        A list of floats representing the audio embedding.
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        # Average over the time dimension to yield a fixed-size vector.
        embedding = np.mean(mel_spec, axis=1)
        return embedding.tolist()
    except Exception:
        logger.exception("Error computing audio embedding.", exc_info=True)
        return []


class Command(BaseCommand):
    """
    Generate embeddings for ContentPost items that lack an embedding.

    This command supports remote storage backends (e.g., S3 via django-storages)
    by opening files via the storage API.
    """

    help = "Generate embeddings for ContentPost items without an existing embedding."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--content_id",
            type=int,
            help=(
                "Optional ContentPost ID to process. "
                "If omitted, process all items without embeddings."
            ),
        )

    def handle(self, *args, **options) -> None:
        queryset = self._get_queryset(options.get("content_id"))
        if not queryset.exists():
            self.stdout.write("No content posts found that require embeddings.")
            return

        total_count = queryset.count()
        self.stdout.write(f"Processing {total_count} content post(s)...")

        for content in queryset:
            embedding = self._process_content(content)
            if embedding:
                content.embedding = embedding
                content.save(update_fields=["embedding"])
                self.stdout.write(
                    f"Embedding computed and saved for ContentPost {content.id}."
                )
            else:
                self.stdout.write(
                    f"Failed to compute embedding for ContentPost {content.id}."
                )

    def _get_queryset(self, content_id: Optional[int]) -> "QuerySet[ContentPost]":
        """
        Retrieve ContentPost queryset filtered by content_id if provided,
        or all posts missing embeddings otherwise.
        """
        if content_id:
            return ContentPost.objects.filter(id=content_id, embedding__isnull=True)
        return ContentPost.objects.filter(embedding__isnull=True)

    def _process_content(self, content: ContentPost) -> List[float]:
        """
        Process a ContentPost instance to compute its embedding.

        Returns:
            The computed embedding as a list of floats, or an empty list if processing fails.
        """
        if content.media_type == "text" and content.text_content:
            self.stdout.write(
                f"Computing text embedding for ContentPost {content.id}..."
            )
            return compute_text_embedding(content.text_content)

        if content.media_type in ("image", "gif") and content.media_file:
            try:
                with content.media_file.open("rb") as file_obj:
                    self.stdout.write(
                        f"Computing image embedding for ContentPost {content.id}..."
                    )
                    return compute_image_embedding_from_file(file_obj)
            except Exception as e:
                self.stdout.write(
                    f"Error processing image for ContentPost {content.id}: {e}"
                )
                return []

        if content.media_type == "audio" and content.media_file:
            return self._process_audio(content)

        if content.media_type == "video":
            self.stdout.write(
                f"Skipping video embedding for ContentPost {content.id} (not implemented)."
            )
            return []

        self.stdout.write(
            f"Insufficient data to compute embedding for ContentPost {content.id}."
        )
        return []

    def _process_audio(self, content: ContentPost) -> List[float] | None:
        """
        Process an audio ContentPost instance by downloading the file to a temporary location.

        Returns:
            The computed audio embedding as a list of floats, or an empty list if processing fails.
        """
        tmp_file_path: Optional[str] = None
        try:
            with content.media_file.open("rb") as file_obj:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3"
                ) as tmp_file:
                    tmp_file.write(file_obj.read())
                    tmp_file_path = tmp_file.name
            self.stdout.write(
                f"Computing audio embedding for ContentPost {content.id}..."
            )
            return compute_audio_embedding(tmp_file_path)
        except Exception as e:
            self.stdout.write(
                f"Error processing audio for ContentPost {content.id}: {e}"
            )
            return []
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)
