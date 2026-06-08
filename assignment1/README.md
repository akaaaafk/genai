# GenAI Assignment 1

This repository contains the implementation for Assignment 1 of the Applied AI / Generative AI course.

The assignment has two main parts:

1. Extending the FastAPI application from the Module 2 class activity.
2. Adding a word embedding endpoint using spaCy.

The API currently supports:

- Text generation using a simple Bigram language model
- Word embedding retrieval using the `en_core_web_lg` spaCy model

## Repository Structure

```text
genai/
├── assignment1/
│   ├── app/
│   │   ├── bigram_model.py
│   │   └── embedding_model.py
│   ├── Assignment1.ipynb
│   └── main.py
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

## Features

### 1. Bigram Text Generation

The project includes a simple Bigram model that learns word transition probabilities from a small corpus.

It can generate text based on:

- A starting word
- A requested output length

### 2. Word Embedding API

The project also includes a word embedding API using spaCy.

The embedding model loads:

```python
en_core_web_lg
```

For a given input word, the API returns:

- The original word
- Whether the word has a vector
- The vector length
- The embedding values as a list

## Requirements

This project uses Python 3.12 or above.

Main dependencies include:

- FastAPI
- Pydantic
- spaCy
- en_core_web_lg
- uvicorn
- uv

The dependencies are managed in `pyproject.toml`.

Assignment1.ipynb contains problem 3

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/akaaaafk/genai.git
cd genai
```

### 2. Install Dependencies

If you are using `uv`, run:

```bash
uv sync
```

### 3. Start the FastAPI Server

```bash
uv run fastapi dev assignment1/main.py
```

After the server starts, open:

```text
http://127.0.0.1:8000/docs
```

This will open the FastAPI Swagger UI, where you can test the API endpoints.

## API Endpoints

### Root Endpoint

```http
GET /
```

Example response:

```json
{
  "message": "FastAPI server is running"
}
```

### Generate Text

```http
POST /generate
```

Request body:

```json
{
  "start_word": "the",
  "length": 10
}
```

Example response:

```json
{
  "generated_text": "the count of monte cristo is a novel written by"
}
```

### Get Word Embedding with POST

```http
POST /embedding
```

Request body:

```json
{
  "word": "king"
}
```

Example response:

```json
{
  "word": "king",
  "has_vector": true,
  "vector_length": 300,
  "embedding": [0.31542, -0.35068, "..."]
}
```

### Get Word Embedding with GET

```http
GET /embedding/{word}
```

Example:

```text
http://127.0.0.1:8000/embedding/king
```

Example response:

```json
{
  "word": "king",
  "has_vector": true,
  "vector_length": 300,
  "embedding": [0.31542, -0.35068, "..."]
}
```

## How to Run with Docker

### 1. Build the Docker Image

```bash
docker build -t sps-genai .
```

### 2. Run the Container

```bash
docker run -p 8000:80 sps-genai
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Notes

The embedding endpoint uses the first token returned by spaCy. Therefore, it is designed mainly for single-word inputs.

For example:

```text
king
computer
student
language
```

If the input word exists in the spaCy model vocabulary, the API returns a 300-dimensional vector.