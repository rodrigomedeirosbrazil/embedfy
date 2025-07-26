import os
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

# Initialize Flask app
app = Flask(__name__)

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
    register_vector(conn)
    return conn

def init_db():
    """Initialize the database tables"""
    conn = get_db_connection()
    cur = conn.cursor()

    # Create table for storing embeddings
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id SERIAL PRIMARY KEY,
            text TEXT NOT NULL,
            embedding VECTOR(384),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    conn.commit()
    cur.close()
    conn.close()

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
            "INSERT INTO embeddings (text, embedding) VALUES (%s, %s) RETURNING id",
            (text, embedding.tolist())
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
            SELECT id, text, embedding <-> %s AS distance
            FROM embeddings
            ORDER BY embedding <-> %s
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

        cur.execute("SELECT id, text, created_at FROM embeddings ORDER BY created_at DESC")
        rows = cur.fetchall()

        texts = []
        for row in rows:
            texts.append({
                "id": row[0],
                "text": row[1],
                "created_at": row[2].isoformat() if row[2] else None
            })

        cur.close()
        conn.close()

        return jsonify({
            "texts": texts,
            "count": len(texts)
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to retrieve texts: {str(e)}"}), 500

if __name__ == '__main__':
    # Initialize database
    init_db()

    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)