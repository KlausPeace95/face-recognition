import cv2 as cv
from my_facerec import MyFaceRec

# Load Camera
cap = cv.VideoCapture(0)

# Encode faces from a folder
mfr = MyFaceRec()
mfr.load_encoding_images("data/trained/")

while True:
    ret, frame = cap.read()
    # Detect Faces
    face_locations, face_names = mfr.detect_known_faces(frame)
    # Display Face Locations
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        cv.putText(frame, name, (x1, y1 - 10), cv.FONT_HERSHEY_DUPLEX, 1, (0,69,255), 2)
        cv.rectangle(frame, (x1, y1), (x2, y2), (0,0,200), 4)

    cv.imshow('Frame', frame)


    key = cv.waitKey(1)
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()


