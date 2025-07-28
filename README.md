# Embedding Microservice

A microservice that receives text prompts via HTTP requests, creates embeddings (vectors) from the text, and stores them in a PostgreSQL vector database.

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

The service uses PostgreSQL with the pgvector extension for persistent vector storage. This allows embeddings to be stored permanently and survive service restarts.

## Docker Support

The service includes Docker support for easy deployment:

1. Build and run with Docker Compose:
   ```bash
   docker-compose up --build
   ```

2. The service will be available at `http://localhost:5000`

3. To stop the services:
   ```bash
   docker-compose down
   ```

### Docker Configuration

The docker-compose.yml file defines two services:
- `app`: The embedding microservice
- `db`: PostgreSQL database with pgvector extension

Environment variables can be modified in the docker-compose.yml file as needed.

## Command Line Interface

The service also includes a command line interface for embedding text files directly:

```bash
python app.py embed-file <path> [--chunk-size N] [--overlap N]
```

The path can be either a text file or a directory. If a directory is specified,
all `.txt` files within the directory and its subdirectories will be processed.

### Arguments

- `path`: Path to the text file or directory to embed (required)
- `--chunk-size`: Size of chunks in characters (default: 500)
- `--overlap`: Overlap between chunks in characters (default: 100)

### Examples

```bash
# Process a single file
python app.py embed-file ./documents/myfile.txt --chunk-size 1000 --overlap 200

# Process all text files in a directory recursively
python app.py embed-file ./documents/ --chunk-size 1000 --overlap 200
```

This will:
- Read the specified text file or all `.txt` files in the directory
- Split each file into chunks of the specified size with overlap
- Generate embeddings for each chunk
- Store the embeddings in the database with metadata (filename, chunk number)

### Metadata

When using the CLI, the following metadata is automatically stored with each embedding:
- `filename`: The name of the source file
- `chunk_number`: The sequential number of the chunk (starting from 1)
- `created_at`: Timestamp when the embedding was created

This metadata can be viewed when retrieving all texts using the `/texts` endpoint.