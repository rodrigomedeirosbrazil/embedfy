# Embedding Microservice

A simple microservice that receives text prompts via HTTP requests, creates embeddings (vectors) from the text, and stores them in a vector database.

## Features

- HTTP API for text embedding
- Vector similarity search
- In-memory vector storage using FAISS
- Health check endpoint

## Requirements

- Python 3.7+
- Flask
- Sentence Transformers
- FAISS
- NumPy

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the server:
   ```bash
   python app.py
   ```

2. The server will start on `http://localhost:5000`

## Testing

To test the microservice, you can use the provided test script:

```bash
python test_microservice.py
```

This script will:
- Check the health of the service
- Create embeddings for sample texts
- Search for similar texts
- Retrieve all stored texts

You can also test the API manually using curl or tools like Postman.

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the service.

### Create Embedding
```
POST /embed
```
Creates an embedding from text and stores it in the vector database.

**Request Body:**
```json
{
  "text": "Your text prompt here"
}
```

**Response:**
```json
{
  "message": "Embedding created and stored successfully",
  "text": "Your text prompt here",
  "embedding_id": 0,
  "embedding_dimension": 384
}
```

### Search Similar Texts
```
POST /search
```
Searches for similar texts using vector similarity.

**Request Body:**
```json
{
  "text": "Your search query here",
  "k": 5  // Number of results to return (optional, default: 5)
}
```

**Response:**
```json
{
  "query": "Your search query here",
  "results": [
    {
      "text": "Similar text from database",
      "distance": 0.123,
      "id": 0
    }
  ]
}
```

### Get All Texts
```
GET /texts
```
Returns all stored texts.

**Response:**
```json
{
  "texts": ["Text 1", "Text 2"],
  "count": 2
}
```

## Model Information

The service uses the `all-MiniLM-L6-v2` model from Sentence Transformers, which creates 384-dimensional embeddings.

## Storage

The service uses FAISS (Facebook AI Similarity Search) for efficient vector similarity search and storage. Note that the current implementation stores vectors in memory, so they will be lost when the service is restarted.