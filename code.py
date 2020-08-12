import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, flash, redirect
from werkzeug.utils import secure_filename

faceDet = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt_tree.xml")

smileDet = cv2.CascadeClassifier("cascades/haarcascade_smile.xml")

faceClassifier = cv2.face.FisherFaceRecognizer_create()
faceClassifier.read("models/FisherFace_model_CK+.xml")

UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = '12345'

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
	return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
	if request.method == 'POST':
		if 'image' not in request.files:
			return 'File not found', 400
	
	image = request.files['image']
	if image.filename == "":
		flash("No File found")
		return redirect(request.url)
	
	if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

	image2 = cv2.imread("uploads/"+image.filename)
	#os.remove(os.path.join(app.config['UPLOAD_FOLDER'], image.filename))
	gray = cv2.cvtColor(image2 ,cv2.COLOR_BGR2GRAY)
	cv2.imshow("image", gray)
	 #Detect face using 4 different classifiers
	face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
	face2 = faceDet2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
	face3 = faceDet3.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)
	face4 = faceDet4.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5), flags=cv2.CASCADE_SCALE_IMAGE)

	#Go over detected faces, stop at first detected face, return empty if no face.
	if len(face) == 1:
		facefeatures = face
	elif len(face2) == 1:
		facefeatures == face2
	elif len(face3) == 1:
		facefeatures = face3
	elif len(face4) == 1:
		facefeatures = face4
	else:
		facefeatures = ""
		#print("face not found")

	#Cut and save face
	for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
		#print("face found")
		grayface = gray[y:y+h, x:x+w] #Cut the frame to size
		resized = cv2.resize(grayface, (350, 350)) #Resize face so all images have same size		
		smile = smileDet.detectMultiScale(resized, 1.1, 10)
		if len(smile) == 1:
			return jsonify({
			'expression': 'happy'
			})
			#print("Expression = happy")
		else:
			expression, file = faceClassifier.predict(resized)
			#print("Expression = ",expression)
			return jsonify({
			'expression': expression
			})

if __name__ == '__main__':
	app.run(port=80, threaded=True)
