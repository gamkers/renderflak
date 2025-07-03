from flask import Flask, request, jsonify
from flask_cors import CORS
import time

app = Flask(__name__)
CORS(app)

messages = []  # Store messages in memory
message_counter = 0  # To track unique message IDs

@app.route('/send_message', methods=['POST'])
def send_message():
    global message_counter
    data = request.json
    message = {
        'id': message_counter,  # Assign unique ID
        'type': data['type'],
        'timestamp': time.time()
    }
    
    if data['type'] == 'text':
        message['message'] = data['message']
    elif data['type'] == 'image':
        message['image'] = data['image']
    
    messages.append(message)
    message_counter += 1  # Increment ID counter
    return jsonify({"status": "success"}), 200

@app.route('/get_messages', methods=['GET'])
def get_messages():
    return jsonify(messages)

if __name__ == '__main__':
    app.run(port=5000)
