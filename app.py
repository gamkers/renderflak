# app.py (Flask API)
from flask import Flask, request, jsonify
from datetime import datetime
import time
from flask_cors import CORS
import uuid

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# In-memory storage for messages
messages = []

@app.route('/messages', methods=['GET', 'POST'])
def handle_messages():
    if request.method == 'POST':
        data = request.json
        message = {
            'id': str(uuid.uuid4()),
            'content': data.get('content', ''),
            'timestamp': int(time.time())
        }
        messages.append(message)
        return jsonify(message), 201
    
    # GET method
    return jsonify(messages)

if __name__ == '__main__':
    app.run(debug=True)
