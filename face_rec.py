import cv2 as cv
import face_recognition as fr

img = cv.imread('data/faces/Tanjoh_Klaus.jpg')
# img0 = cv.resize(img, (500,500), interpolation=cv.INTER_AREA)
rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_encoding = fr.face_encodings(rgb_img)[0]

img1 = cv.imread('data/faces/Will_Jesse.jpg')
img2 = cv.resize(img1, (500,500), interpolation=cv.INTER_AREA)
rgb_img2 = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img_encoding2 = fr.face_encodings(rgb_img2)[0]

result = fr.compare_faces([img_encoding], img_encoding2)
print("Result: ", result)

cv.imshow('Klaus', img)
cv.imshow('Jesse', img2)

cv.waitKey(0)