import logging
import os
import subprocess
import tempfile
from typing import List, Optional, BinaryIO

import librosa
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from faster_whisper import WhisperModel
from transformers import AutoTokenizer, AutoModel

from core.models import ContentPost

logger = logging.getLogger(__name__)

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddingProcessor:
    """
    Processes ContentPost instances to compute embeddings based on media type.
    Supports text, image, audio, and video (via audio extraction and transcription).
    """

    def __init__(self) -> None:
        self.text_tokenizer: Optional[AutoTokenizer] = None
        self.text_model: Optional[AutoModel] = None
        self.image_model: Optional[torch.nn.Module] = None
        self.image_transform: Optional[transforms.Compose] = None

    def init_text_model(self) -> None:
        if self.text_model is None:
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                "distilbert-base-uncased"
            )
            self.text_model = AutoModel.from_pretrained("distilbert-base-uncased")
            self.text_model.to(DEVICE)
            self.text_model.eval()

    def compute_text_embedding(self, text: str) -> List[float]:
        self.init_text_model()
        inputs = self.text_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.text_model(**inputs)
        # Mean-pool over token embeddings.
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        return embedding.tolist()

    def init_image_model(self) -> None:
        if self.image_model is None:
            base_model = models.resnet18(pretrained=True)
            # Remove the classification head.
            modules = list(base_model.children())[:-1]
            self.image_model = torch.nn.Sequential(*modules)
            self.image_model.to(DEVICE)
            self.image_model.eval()
            self.image_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )

    def compute_image_embedding(self, file_obj: BinaryIO) -> List[float]:
        self.init_image_model()
        image = Image.open(file_obj).convert("RGB")
        image_tensor = self.image_transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = self.image_model(image_tensor).squeeze().cpu().numpy()
        return embedding.flatten().tolist()

    def compute_audio_embedding(self, audio_path: str) -> List[float]:
        try:
            y, sr = librosa.load(audio_path, sr=22050)
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            embedding = np.mean(mel_spec, axis=1)
            return embedding.tolist()
        except Exception:
            logger.exception("Error computing audio embedding.", exc_info=True)
            return []

    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe the audio at the given path using faster-whisper.

        Args:
            audio_path: Path to the audio file.

        Returns:
            The transcribed text, or an empty string if transcription fails.
        """
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = WhisperModel("base", device=device, compute_type="float32")
            segments, info = model.transcribe(
                audio_path, beam_size=5, word_timestamps=False
            )
            transcript = " ".join(segment.text for segment in segments)
            return transcript.strip()
        except Exception:
            logger.exception(
                "Error transcribing audio with faster-whisper.", exc_info=True
            )
            return ""

    def process_text(self, content: ContentPost) -> List[float]:
        logger.info(f"Computing text embedding for ContentPost {content.id}...")
        return self.compute_text_embedding(content.text_content)

    def process_image(self, content: ContentPost) -> List[float]:
        try:
            with content.media_file.open("rb") as file_obj:
                logger.info(
                    f"Computing image embedding for ContentPost {content.id}..."
                )
                return self.compute_image_embedding(file_obj)
        except Exception as e:
            logger.error(f"Error processing image for ContentPost {content.id}: {e}")
            return []

    def process_audio(self, content: ContentPost) -> List[float]:
        """
        Process an audio post by attempting to transcribe the audio.
        If transcription yields a non-empty transcript, return its text embedding;
        otherwise, fall back to the audio embedding.
        """
        tmp_file_path: Optional[str] = None
        try:
            with content.media_file.open("rb") as file_obj:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3"
                ) as tmp_file:
                    tmp_file.write(file_obj.read())
                    tmp_file_path = tmp_file.name

            logger.info(f"Transcribing audio for ContentPost {content.id}...")
            transcript = self.transcribe_audio(tmp_file_path)
            if transcript:
                logger.info(f"Transcription successful for ContentPost {content.id}.")
                return self.compute_text_embedding(transcript)
            else:
                logger.info(
                    f"No speech found for ContentPost {content.id}, falling back to audio embedding."
                )
                return self.compute_audio_embedding(tmp_file_path)
        except Exception as e:
            logger.error(f"Error processing audio for ContentPost {content.id}: {e}")
            return []
        finally:
            if tmp_file_path and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    def process_video(self, content: ContentPost) -> List[float]:
        """
        Process a video post by extracting its audio, then attempting transcription.
        If transcription yields text, return its text embedding; otherwise, fall back to audio embedding.
        """
        tmp_video_path: Optional[str] = None
        tmp_audio_path: Optional[str] = None
        try:
            with content.media_file.open("rb") as file_obj:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp4"
                ) as tmp_video_file:
                    tmp_video_file.write(file_obj.read())
                    tmp_video_path = tmp_video_file.name

            with tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp3"
            ) as tmp_audio_file:
                tmp_audio_path = tmp_audio_file.name

            logger.info(f"Extracting audio from video for ContentPost {content.id}...")
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    tmp_video_path,
                    "-vn",
                    "-acodec",
                    "libmp3lame",
                    "-q:a",
                    "2",
                    tmp_audio_path,
                ],
                capture_output=True,
            )
            if result.returncode != 0:
                logger.error(
                    f"FFmpeg error for ContentPost {content.id}: {result.stderr.decode()}"
                )
                return []
            logger.info(f"Transcribing video audio for ContentPost {content.id}...")
            transcript = self.transcribe_audio(tmp_audio_path)
            if transcript:
                logger.info(f"Transcription successful for ContentPost {content.id}.")
                return self.compute_text_embedding(transcript)
            else:
                logger.info(
                    f"No speech found for ContentPost {content.id}, falling back to audio embedding."
                )
                return self.compute_audio_embedding(tmp_audio_path)
        except Exception as e:
            logger.error(f"Error processing video for ContentPost {content.id}: {e}")
            return []
        finally:
            if tmp_audio_path and os.path.exists(tmp_audio_path):
                os.remove(tmp_audio_path)
            if tmp_video_path and os.path.exists(tmp_video_path):
                os.remove(tmp_video_path)

    def process_content(self, content: ContentPost) -> List[float]:
        """
        Process a ContentPost instance according to its media type.
        """
        if content.media_type == "text" and content.text_content:
            return self.process_text(content)
        if content.media_type in ("image", "gif") and content.media_file:
            return self.process_image(content)
        if content.media_type == "audio" and content.media_file:
            return self.process_audio(content)
        if content.media_type == "video" and content.media_file:
            return self.process_video(content)

        logger.info(
            f"Insufficient data to compute embedding for ContentPost {content.id}."
        )
        return []
