import logging
import os
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

# Global variables for models and tokenizers
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
text_tokenizer: Optional[AutoTokenizer] = None
text_model: Optional[AutoModel] = None
image_model: Optional[torch.nn.Module] = None
image_transform: Optional[transforms.Compose] = None


def init_text_model() -> None:
    """
    Initialize the text model and tokenizer if not already done.
    """
    global text_tokenizer, text_model
    if text_model is None:
        text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        text_model = AutoModel.from_pretrained("distilbert-base-uncased")
        text_model.to(device)
        text_model.eval()


def compute_text_embedding(text: str) -> List[float]:
    """
    Compute a text embedding using a transformer model.

    Args:
        text: The input text to embed.

    Returns:
        A list of floats representing the text embedding.
    """
    init_text_model()
    inputs = text_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    # Move inputs to the same device as the model.
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = text_model(**inputs)
    # Mean pooling over token embeddings.
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embedding.tolist()


def init_image_model() -> None:
    """
    Initialize the image model and transformation pipeline if not already done.
    """
    global image_model, image_transform
    if image_model is None:
        base_model = models.resnet18(pretrained=True)
        # Remove the final classification layer.
        modules = list(base_model.children())[:-1]
        image_model = torch.nn.Sequential(*modules)
        image_model.to(device)
        image_model.eval()
        image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def compute_image_embedding(image_path: str) -> List[float]:
    """
    Compute an image embedding using a pre-trained CNN.

    Args:
        image_path: Path to the image file.

    Returns:
        A list of floats representing the image embedding.
    """
    init_image_model()
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = image_model(image_tensor).squeeze().cpu().numpy()
    # Flatten the embedding in case it is multi-dimensional.
    return embedding.flatten().tolist()


def compute_audio_embedding(audio_path: str) -> List[float]:
    """
    Compute an audio embedding using a mel-spectrogram based approach.

    This is a placeholder implementation; in production, you might replace it
    with an audio-specific model such as VGGish.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A list of floats representing the audio embedding.
    """
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        # Average over the time dimension to obtain a fixed-size vector.
        embedding = np.mean(mel_spec, axis=1)
        return embedding.tolist()
    except Exception:
        logger.exception("Error computing audio embedding.", exc_info=True)
        return []


class Command(BaseCommand):
    """
    Management command to generate embeddings for ContentPost items that lack an embedding.
    """

    help = "Generate embeddings for ContentPost items without an existing embedding."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--content_id",
            type=int,
            help="Optional ContentPost ID to process. If omitted, all items without embeddings are processed.",
        )

    def handle(self, *args, **options) -> None:
        content_id: Optional[int] = options.get("content_id")
        if content_id:
            queryset = ContentPost.objects.filter(id=content_id, embedding__isnull=True)
        else:
            queryset = ContentPost.objects.filter(embedding__isnull=True)

        if not queryset.exists():
            self.stdout.write("No content posts found that require embeddings.")
            return

        total_count: int = queryset.count()
        self.stdout.write(f"Processing {total_count} content post(s)...")

        for content in queryset:
            embedding: List[float] = []

            if content.media_type == "text" and content.text_content:
                self.stdout.write(
                    f"Computing text embedding for ContentPost {content.id}..."
                )
                embedding = compute_text_embedding(content.text_content)
            elif content.media_type in ("image", "gif") and content.media_file:
                file_path: str = content.media_file.path
                if os.path.exists(file_path):
                    self.stdout.write(
                        f"Computing image embedding for ContentPost {content.id}..."
                    )
                    embedding = compute_image_embedding(file_path)
                else:
                    self.stdout.write(
                        f"File not found for ContentPost {content.id}: {file_path}"
                    )
            elif content.media_type == "audio" and content.media_file:
                file_path = content.media_file.path
                if os.path.exists(file_path):
                    self.stdout.write(
                        f"Computing audio embedding for ContentPost {content.id}..."
                    )
                    embedding = compute_audio_embedding(file_path)
                else:
                    self.stdout.write(
                        f"File not found for ContentPost {content.id}: {file_path}"
                    )
            elif content.media_type == "video":
                self.stdout.write(
                    f"Skipping video embedding for ContentPost {content.id} (not implemented)."
                )
                continue
            else:
                self.stdout.write(
                    f"Insufficient data to compute embedding for ContentPost {content.id}."
                )
                continue

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
