from django.core.exceptions import ValidationError
import magic


def validate_media_file(file):
    """Validate that a media file is not larger than 10MB and is an allowed type."""
    # Check file size
    max_size = 30 * 1024 * 1024  # 30MB
    if file.size > max_size:
        raise ValidationError("File size should not exceed 30MB.")

    # Read a sample of the file for MIME detection.
    file.seek(0)
    file_header = file.read(2048)
    file.seek(0)  # reset file pointer

    mime = magic.Magic(mime=True)
    mime_type = mime.from_buffer(file_header)

    allowed_mime_types = [
        # Images
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        # Videos
        "video/mp4",
        "video/quicktime",  # commonly .mov
        # Audio
        "audio/mpeg",
        "audio/wav",
        "audio/x-wav",
        "audio/ogg",
    ]

    if mime_type not in allowed_mime_types:
        raise ValidationError(
            f"Unsupported file type: {mime_type}. Allowed types are JPEG, PNG, GIF, WEBP, MP4, MOV, MP3, WAV, and OGG."
        )
