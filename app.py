import cv2
import boto3
import base64
import numpy as np
import os
import random
from flask import Flask, render_template, Response, request, jsonify, send_from_directory

app = Flask(__name__)

rekognition_client = boto3.client('rekognition')

# Mapping of emotions to directories
emotion_to_directory = {
    'HAPPY': 'happy',
    'SAD': 'sad',
    'ANGRY': 'angry',
    'SURPRISED': 'surprised',
    'DISGUSTED': 'disgusted',
    'CALM': 'calm'
}

@app.route('/')
def index():
    return render_template('index.html')

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture():
    image_data = request.json.get('image')
    if image_data is None:
        return jsonify({'error': 'No image data received'}), 400

    image_data = base64.b64decode(image_data.split(',')[1])
    image = np.frombuffer(image_data, dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    _, jpeg_image = cv2.imencode('.jpg', image)
    
    response = rekognition_client.detect_faces(
        Image={'Bytes': jpeg_image.tobytes()},
        Attributes=['ALL']
    )
    
    emotions = response['FaceDetails'][0]['Emotions']
    primary_emotion = max(emotions, key=lambda x: x['Confidence'])['Type']
    
    # Get the directory for the primary emotion
    emotion_directory = emotion_to_directory.get(primary_emotion, 'calm')
    
    # Select a random song from the directory
    songs_dir = os.path.join('songs', emotion_directory)
    song = random.choice(os.listdir(songs_dir))
    song_url = f'/songs/{emotion_directory}/{song}'
    
    return jsonify({'emotion': primary_emotion, 'song_url': song_url})

@app.route('/songs/<emotion>/<filename>')
def serve_song(emotion, filename):
    return send_from_directory(os.path.join('songs', emotion), filename)

if __name__ == "__main__":
    app.run(debug=True)
