import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Database connection parameters
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_NAME = os.environ.get('DB_NAME', 'embeddings_db')
DB_USER = os.environ.get('DB_USER', 'postgres')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'postgres')
DB_PORT = os.environ.get('DB_PORT', '5432')

def get_db_connection():
    """Create a database connection"""
    conn = psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        port=DB_PORT
    )
    return conn

def init_db():
    """Initialize the database tables"""
    conn = get_db_connection()
    cur = conn.cursor()

    # Enable pgvector extension
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.commit()
        # Now register the vector type after creating the extension
        register_vector(conn)
    except Exception as e:
        print(f"Warning: Could not create vector extension: {e}")
        conn.rollback()
        cur.close()
        conn.close()
        return

    # Create table for storing embeddings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding VECTOR(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            chunk_number INTEGER
        )
    """)

    # Add metadata columns if they don't exist (for backward compatibility)
    try:
        cur.execute("ALTER TABLE embeddings ADD COLUMN filename TEXT")
        conn.commit()
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()

    try:
        cur.execute("ALTER TABLE embeddings ADD COLUMN chunk_number INTEGER")
        conn.commit()
    except psycopg2.errors.DuplicateColumn:
        conn.rollback()

    conn.commit()
    cur.close()
    conn.close()

def chunk_text(text, chunk_size=500, overlap=100):
    """Split text into chunks with overlap"""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)

        # If we've reached the end of the text, break
        if end == len(text):
            break

        # Move start position for next chunk with overlap
        start = end - overlap

        # Ensure we don't get stuck in an infinite loop
        if start >= len(text):
            break

    return chunks

def embed_file(file_path, chunk_size=500, overlap=100):
    """Embed a text file by splitting it into chunks and storing embeddings with metadata"""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Error: File '{file_path}' not found")
            return False

        # Read the file
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()

        if not text.strip():
            print(f"Error: File '{file_path}' is empty")
            return False

        # Split text into chunks
        chunks = chunk_text(text, chunk_size, overlap)
        print(f"Split file into {len(chunks)} chunks")

        # Generate embeddings for all chunks
        embeddings = model.encode(chunks)

        # Store embeddings with metadata
        conn = get_db_connection()
        cur = conn.cursor()

        success_count = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            try:
                cur.execute(
                    "INSERT INTO embeddings (text, embedding, filename, chunk_number) VALUES (%s, %s, %s, %s) RETURNING id",
                    (chunk, embedding.tolist(), os.path.basename(file_path), i+1)
                )
                cur.fetchone()
                success_count += 1
            except Exception as e:
                print(f"Warning: Failed to store chunk {i+1}: {str(e)}")

        conn.commit()
        cur.close()
        conn.close()

        print(f"Successfully embedded {success_count} of {len(chunks)} chunks from '{file_path}'")
        return True

    except Exception as e:
        print(f"Error embedding file '{file_path}': {str(e)}")
        return False

def embed_directory(dir_path, chunk_size=500, overlap=100):
    """Recursively embed all text files in a directory"""
    try:
        # Check if directory exists
        if not os.path.exists(dir_path):
            print(f"Error: Directory '{dir_path}' not found")
            return False

        if not os.path.isdir(dir_path):
            print(f"Error: '{dir_path}' is not a directory")
            return False

        # Process all text files recursively
        total_files = 0
        successful_files = 0

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                # Process only text files
                if file.endswith('.txt') or file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    total_files += 1
                    if embed_file(file_path, chunk_size, overlap):
                        successful_files += 1

        print(f"Processed {successful_files} of {total_files} text files in directory '{dir_path}'")
        return successful_files == total_files

    except Exception as e:
        print(f"Error embedding directory '{dir_path}': {str(e)}")
        return False

if __name__ == '__main__':
    import sys
    import argparse

    # Check if we're running in CLI mode
    if len(sys.argv) > 1 and sys.argv[1] == 'embed-file':
        # CLI mode for embedding files or directories
        parser = argparse.ArgumentParser(description='Embed text files into the vector database')
        parser.add_argument('command', help='Command to run (embed-file)')
        parser.add_argument('path', help='Path to the text file or directory to embed')
        parser.add_argument('--chunk-size', type=int, default=500, help='Size of chunks in characters (default: 500)')
        parser.add_argument('--overlap', type=int, default=100, help='Overlap between chunks in characters (default: 100)')

        args = parser.parse_args()

        # Initialize database
        init_db()

        # Check if path is a file or directory and process accordingly
        if os.path.isfile(args.path):
            success = embed_file(args.path, args.chunk_size, args.overlap)
        elif os.path.isdir(args.path):
            success = embed_directory(args.path, args.chunk_size, args.overlap)
        else:
            print(f"Error: Path '{args.path}' does not exist")
            success = False

        sys.exit(0 if success else 1)
    else:
        # Web server mode (default)
        # Initialize Flask app
        app = Flask(__name__)
        from flask import request, jsonify

        # Register routes
        @app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            try:
                conn = get_db_connection()
                conn.close()
                return jsonify({"status": "healthy"}), 200
            except Exception as e:
                return jsonify({"status": "unhealthy", "error": str(e)}), 500

        @app.route('/embed', methods=['POST'])
        def create_embedding():
            """Create embedding from text and store in vector database"""
            try:
                # Get the text from request
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({"error": "Missing 'text' in request body"}), 400

                text = data['text']
                if not text.strip():
                    return jsonify({"error": "Text cannot be empty"}), 400

                # Generate embedding
                embedding = model.encode([text])[0]

                # Store in database
                conn = get_db_connection()
                cur = conn.cursor()

                cur.execute(
                    "INSERT INTO embeddings (text, embedding, filename, chunk_number) VALUES (%s, %s, %s, %s) RETURNING id",
                    (text, embedding.tolist(), None, None)
                )

                embedding_id = cur.fetchone()[0]
                conn.commit()
                cur.close()
                conn.close()

                # Return success response with embedding info
                return jsonify({
                    "message": "Embedding created and stored successfully",
                    "text": text,
                    "embedding_id": embedding_id,
                    "embedding_dimension": len(embedding)
                }), 201

            except Exception as e:
                return jsonify({"error": f"Failed to create embedding: {str(e)}"}), 500

        @app.route('/search', methods=['POST'])
        def search_similar():
            """Search for similar texts using vector similarity"""
            try:
                # Get the text from request
                data = request.get_json()
                if not data or 'text' not in data:
                    return jsonify({"error": "Missing 'text' in request body"}), 400

                text = data['text']
                if not text.strip():
                    return jsonify({"error": "Text cannot be empty"}), 400

                # Get number of results to return (default to 5)
                k = data.get('k', 5)

                # Generate embedding for search query
                query_embedding = model.encode([text])[0]

                # Search in database
                conn = get_db_connection()
                cur = conn.cursor()

                cur.execute(
                    """
                    SELECT id, text, embedding <-> %s::vector AS distance
                    FROM embeddings
                    ORDER BY embedding <-> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding.tolist(), query_embedding.tolist(), k)
                )

                results = []
                for row in cur.fetchall():
                    results.append({
                        "id": row[0],
                        "text": row[1],
                        "distance": float(row[2])
                    })

                cur.close()
                conn.close()

                return jsonify({
                    "query": text,
                    "results": results
                }), 200

            except Exception as e:
                return jsonify({"error": f"Failed to search: {str(e)}"}), 500

        @app.route('/texts', methods=['GET'])
        def get_all_texts():
            """Get all stored texts"""
            try:
                conn = get_db_connection()
                cur = conn.cursor()

                cur.execute("SELECT id, text, created_at, filename, chunk_number FROM embeddings ORDER BY created_at DESC")
                rows = cur.fetchall()

                texts = []
                for row in rows:
                    texts.append({
                        "id": row[0],
                        "text": row[1],
                        "created_at": row[2].isoformat() if row[2] else None,
                        "filename": row[3],
                        "chunk_number": row[4]
                    })

                cur.close()
                conn.close()

                return jsonify({
                    "texts": texts,
                    "count": len(texts)
                }), 200

            except Exception as e:
                return jsonify({"error": f"Failed to retrieve texts: {str(e)}"}), 500

        # Initialize database
        init_db()

        # Run the app
        app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)