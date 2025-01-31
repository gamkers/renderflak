from flask import Flask, request, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

messages = []  # Store messages in memory

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    message = {
        'type': data['type'],
        'timestamp': time.time()
    }
    
    if data['type'] == 'text':
        message['message'] = data['message']
    elif data['type'] == 'image':
        message['image'] = data['image']
    
    messages.append(message)
    return jsonify({"status": "success"}), 200

@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify(messages)

if __name__ == '__main__':
    app.run(port=5000)
