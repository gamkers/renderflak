from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import time
import uuid

app = Flask(__name__)
CORS(app)

# Store messages in memory
messages = []

@app.route('/message', methods=['POST'])
def receive_message():
    try:
        message_id = str(uuid.uuid4())
        timestamp = time.time()
        
        if 'image' in request.files:  # Image message
            image_file = request.files['image']
            image_bytes = image_file.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            
            message = {
                'id': message_id,
                'type': 'image',
                'timestamp': timestamp,
                'image': base64_image,
                'filename': image_file.filename,
                'text': request.form.get('text', '')  # Optional text with image
            }
        else:  # Text message
            data = request.json
            message = {
                'id': message_id,
                'type': 'text',
                'timestamp': timestamp,
                'text': data.get('text', ''),
                'sender': data.get('sender', 'Unknown')
            }
        
        messages.append(message)
        return jsonify({'message': 'Message received', 'id': message_id}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/messages', methods=['GET'])
def get_messages():
    return jsonify(messages)

if __name__ == '__main__':
    app.run(debug=True)
