![Logo](core/static/core/img/strvsocial-logo.png)


Read the docs: https://www.dropbox.com/scl/fi/c3uw71qq0zcun7tl415p9/STRV.social.pdf?rlkey=gu7sxkbusnrssf1vltbx1jh1f&dl=0


## Overview
This repository contains a prototype solution designed to find and present content similar to user uploads for a social
network inspired by Instagram and Pinterest. The prototype efficiently manages multiple media types and large amounts of content items using embeddings
to enable rapid similarity searches.

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)

---

## Project Purpose
The main goal is to build a highly scalable and efficient solution to help users easily discover the most relevant
and similar content from potentially millions of items. This prototype demonstrates the feasibility and efficiency of
content similarity search leveraging modern AI techniques.

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)

## Features

- **Multi-media Embeddings:** Supports text, images, and audio content embedding generation.
- **Efficient Retrieval:** Quickly retrieves relevant content through similarity search.
- **Scalable:** Designed for scalability, capable of handling millions of content items.
- **Integration Ready:** Easily deployable via Docker containers.

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)

## Technologies Used

- **Backend:** Django, PostgreSQL
- **AI & ML Libraries:** PyTorch, Transformers, torchvision, librosa
- **Frontend:** HTMX, Tailwind CSS, Font Awesome
- **Deployment:** Docker, Docker Compose

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)

## How It Works

The application generates embeddings using pre-trained AI models:
- **Text:** DistilBERT for semantic understanding.
- **Images:** ResNet18 CNN model for visual embeddings.
- **Audio:** Mel-spectrogram-based embeddings.

Embeddings enable rapid similarity searches using vector databases or similarity metrics (e.g., cosine similarity) optimized for scalability and speed.

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)

## Getting Started

### Prerequisites
- Docker & Docker Compose installed

## Installation & Setup

Clone the repository:

```bash
git clone https://github.com/your-repo-url
cd your-repo
```

Build and run with Docker Compose:

```bash
docker-compose build
docker-compose up
```

The server will run locally at `http://localhost:8000`

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)

## Usage

### Generate Embeddings
Generate embeddings for all items lacking them:

```bash
docker-compose run --rm django python manage.py genembeddings
```

Generate embeddings for specific content ID:

```bash
docker-compose run django python manage.py genembeddings --content_id=123
```

This is done periodically in production to keep embeddings up-to-date.

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)

## Deployment

- Embeddings are generated asynchronously and stored in the database.
- Use vector databases (e.g., FAISS, Pinecone) for efficient similarity searches in production.
- Employ Kubernetes for scaling horizontally, with autoscaling workers for embedding tasks.

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)

## Turning into a Recommendation Engine

Future enhancements:

- Implement user interaction feedback for personalization.
- Integrate collaborative filtering alongside content-based filtering.
- Leverage user engagement data to fine-tune recommendations.

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)

## Next Steps

- Extend embedding generation support for video content.
- Integrate a vector search engine for production-grade scalability.
- Build a recommendation system based on user interactions.

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)


## License

This project is licensed under the MIT License.

[⬆️ Back to Top](#strv-social-network-content-similarity-prototype)


