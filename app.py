from flask import Flask, request, jsonify
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Initialize Flask app
app = Flask(__name__)

# Initialize the sentence transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index for vector storage
dimension = 384  # Dimension of the embeddings from all-MiniLM-L6-v2
index = faiss.IndexFlatL2(dimension)

# In-memory storage for text prompts (for demonstration)
text_storage = []

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy"}), 200

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
        embedding = model.encode([text])

        # Store in FAISS index
        index.add(embedding.astype('float32'))

        # Store the original text (for demonstration)
        text_storage.append(text)

        # Return success response with embedding info
        return jsonify({
            "message": "Embedding created and stored successfully",
            "text": text,
            "embedding_id": len(text_storage) - 1,
            "embedding_dimension": embedding.shape[1]
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
        query_embedding = model.encode([text])

        # Search in FAISS index
        distances, indices = index.search(query_embedding.astype('float32'), k)

        # Prepare results
        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx < len(text_storage):  # Check if index is valid
                results.append({
                    "text": text_storage[idx],
                    "distance": float(distances[0][i]),
                    "id": int(idx)
                })

        return jsonify({
            "query": text,
            "results": results
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to search: {str(e)}"}), 500

@app.route('/texts', methods=['GET'])
def get_all_texts():
    """Get all stored texts"""
    return jsonify({
        "texts": text_storage,
        "count": len(text_storage)
    }), 200

if __name__ == '__main__':
    # Run the app
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)