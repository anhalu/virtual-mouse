import os
from Hand_Tracking_Module_cvzone_custom import HandDetector
import cv2
from tensorflow.keras.models import load_model

instance_bonus = 50
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.7, maxHands=1)
model = load_model('/home/halu/anhalu-data/Deep Learning/Final_model/MobileNet_with_new_data.h5')
name_gesture = ['Double_Left_mouse',
 'Left_mouse',
 'Long_press_mouse',
 'Move_mouse',
 'Right_mouse',
 'Scroll_mouse']

while True:
    success, img_cam = cap.read()

    img = cv2.flip(img_cam, 1)
    
    hands, img_draw = detector.findHands(img.copy(), draw=True, flipType=False)

    if hands:
        hands1 = hands[0]
        bbox1 = hands1['bbox']
        img_crop = img[bbox1[1]-instance_bonus:bbox1[1]+bbox1[3]+instance_bonus, 
                       bbox1[0]-instance_bonus:bbox1[0]+bbox1[2]+instance_bonus]
        try:
            X = cv2.resize(img_crop, (224, 224))
            X = X.reshape((1,) + X.shape)
            X = X / 255.0
            y = model.predict(X)
            print(y)
            id = y.argmax(axis=1)
            print(name_gesture[id[0]])
            cv2.putText(img_draw, name_gesture[id[0]], (bbox1[0], bbox1[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        except Exception as e:
            print(e)
    cv2.imshow('cap', img_draw)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()