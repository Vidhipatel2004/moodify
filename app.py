import base64
from flask import Flask, render_template, request, jsonify, redirect, url_for
import cv2
import numpy as np
import webbrowser
from inference import perform_inference


app = Flask(__name__)


# Dummy user database (replace with actual database)
users = {
    'user1@example.com': 'password1',
    'user2@example.com': 'password2'
}

# Route for serving the registration page
@app.route('/', methods=['GET'])
def register_page():
    return render_template('register.html')

# Route for handling user registration
@app.route('/', methods=['POST'])
def register():
    email = request.form['email']
    password = request.form['password']

    # Check if user already exists
    if email in users:
        return render_template('.html', error="Email already exists")

    # Add new user to database (in this case, just a dictionary)
    users[email] = password

    # Perform ML operations if needed
    # prediction = model.predict(email, password)

    return render_template('land.html')
    

# Route for serving the login page
@app.route('/login', methods=['GET'])
def login_page():
    return render_template('login.html')

# Route for handling user login
@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    # Check if user exists and password is correct
    if email in users and users[email] == password:
        # Perform ML operations if needed
        # prediction = model.predict(email, password)

        return render_template('land.html')
    else:
        return render_template('login.html', error="Invalid Username or Password")


# Route for serving the landing page


@app.route('/land')
def landing_page():
    return render_template('land.html')

# Route for serving the inference page
@app.route('/inference')
def inference_page():
    return render_template('inference.html')

# Route for capturing the face and performing inference
@app.route('/capture-face', methods=['POST'])
def capture_face():
    # Assuming the image data is sent as a base64-encoded string in the request body
    image_data = request.json['image_data']
    
    # Convert base64 image data to OpenCV image
    nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Perform inference to detect emotion
    emotion_prediction = perform_inference(image)
    
    # Redirect to Spotify based on the detected emotion
    return redirect(url_for('recommend_songs', emotion=emotion_prediction))


   
   # Route for recommending songs based on detected emotion
@app.route('/recommend-songs/<emotion>', methods=['GET'])
def recommend_songs(emotion):
    # Dictionary mapping emotions to Spotify playlist URLs
    playlist_urls = {
        'happy': 'https://open.spotify.com/playlist/4l7S8jYce53k5qriDma6Ha?si=8b0334225ade4ca8',
        'sad': 'https://open.spotify.com/playlist/06UM7loEdJCss4orDOb4XJ?si=acc8328860534a5b',
        'chill': 'https://open.spotify.com/playlist/4j4grVbkouSjZu87VNhhIn?si=554f885981274f15',
        'motivated':'https://open.spotify.com/playlist/4dVo5D78Wnnw9vlFUSLyQD?si=976521c9f6a54e9c',
        'rock':'https://open.spotify.com/playlist/0H6xiop1gBSko5uGdNrRqA?si=6f8fcd68e7f94ef1'
        # Add more emotions and corresponding playlist URLs as needed
    }
    
    # Check if the detected emotion has a corresponding playlist URL
    if emotion.lower() in playlist_urls:
        # Redirect to the corresponding playlist URL
        webbrowser.open_new_tab(playlist_urls[emotion.lower()])
    else:
        # If no specific playlist URL is found, redirect to the landing page
        return redirect(url_for('landing_page'))


if __name__ == '__main__':
    app.run(debug=True)
