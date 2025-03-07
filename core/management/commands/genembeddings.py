from typing import Optional

from django.core.management.base import BaseCommand

from core.models import ContentPost
from core.processor import EmbeddingProcessor


class Command(BaseCommand):
    """
    Generate embeddings for ContentPost items that lack an embedding.
    This command supports remote storage by using the file-like API.
    """

    help = "Generate embeddings for ContentPost items without an existing embedding."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--content_id",
            type=int,
            help="Optional ContentPost ID to process. If omitted, process all items without embeddings.",
        )

    def handle(self, *args, **options) -> None:
        queryset = self._get_queryset(options.get("content_id"))
        if not queryset.exists():
            self.stdout.write("No content posts found that require embeddings.")
            return

        total_count = queryset.count()
        self.stdout.write(f"Processing {total_count} content post(s)...")

        processor = EmbeddingProcessor()
        for content in queryset:
            embedding = processor.process_content(content)
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

    def _get_queryset(self, content_id: Optional[int]):
        if content_id:
            return ContentPost.objects.filter(id=content_id, embedding__isnull=True)
        return ContentPost.objects.filter(embedding__isnull=True)
