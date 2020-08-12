import cv2

faceDet = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
faceDet2 = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
faceDet3 = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
faceDet4 = cv2.CascadeClassifier("haarcascade_frontalface_alt_tree.xml")

smileDet = cv2.CascadeClassifier("haarcascade_smile.xml")

faceClassifier = cv2.face.FisherFaceRecognizer_create()
faceClassifier.read("models/FisherFace_model_CK+.xml")
image = cv2.imread("images/happy4.png")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

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
    print("face not found")

#Cut and save face
for (x, y, w, h) in facefeatures: #get coordinates and size of rectangle containing face
    print("face found")
    grayface = gray[y:y+h, x:x+w] #Cut the frame to size
    resized = cv2.resize(grayface, (350, 350)) #Resize face so all images have same size
    cv2.imshow("resized",resized)
    cv2.waitKey()
    smile = smileDet.detectMultiScale(resized, 1.1, 10)
    if len(smile) == 1:
        print("Happy expression")
    else:
        expression, file = faceClassifier.predict(resized)
        print("Expression = ",expression)
        print("file = ",file)