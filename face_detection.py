#import openCv module
import cv2 as cv
#creating a cascade classifier object, used haarcascade_frontal_face classifier
face_cascade = cv.CascadeClassifier("haarcascade_frontalface.xml")

#reading image
#add your image path as the argument here.
img = cv.imread("mypic.jpg")

#reading image as grayscale image
gray_img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#we are converting to gray scale because, it require less information to be provided for each pixel henceforth, it is good practice to
#convert color img to grayscale.

faces = face_cascade.detectMultiScale(gray_img,scaleFactor = 1.05, minNeighbors = 5)
#here, detectMultiScale is a method to search for face rectangle coords.
#scaleFactor, decreases the shape value by 5% until the face is found. Smaller this value, Greater is the efficiency.

#adding rectangle box to the face
for x,y,w,h in faces :
    img = cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 3)
#rectangle is a method to create rectangular box
#2,3,4 arguments are the RGB values of rectangular outline
#last argument is the width of the rectangle.

#to show the image
cv.imshow("Image", img)
#wait until we press any button, 0 means 0 milliseconds
cv.waitKey(0)
#finally, destroys all active windows
cv.destroyAllWindows()

#Contributed by Gopesh Kosala
#Feb 16, 2021
