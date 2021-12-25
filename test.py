import cv2
import mediapipe as mp
import numpy as np
import math
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def blurr(img,x,y):
    p1 = (x, y)
    w, h = 100, 100
    p2 = (p1[0] + w, p1[1] + h)


    circle_center = ((p1[0] + p2[0])// 2, (p1[1] + p2[1]) // 2)
    circle_radius = int(math.sqrt(w * w + h * h) // 2)
    mask_img = np.zeros(img.shape, dtype='uint8')
    cv2.circle(mask_img,(x,y),50, (255, 255, 255), -1)#circle_center, circle_radius, (255, 255, 255), -1)

    img_all_blurred = cv2.medianBlur(img, 99)
    img_face_blurred = np.where(mask_img > 0, img_all_blurred, img)
    return img_face_blurred
    # VIDEO FEED
"""
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Mediapipe Feed', frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
"""
cap = cv2.VideoCapture("output1.avi")
## Setup mediapipe instance
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output2.avi', fourcc, 20.0, (640,  480))
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if frame is not None:
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
        
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            a=results.pose_landmarks
            coordinate=a.landmark[0]
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    )
            h,w,c=image.shape                                       
            #cv2.circle(image, (int(coordinate.x*w), int(coordinate.y*h)), 50, (255,0,0))
            image=blurr(image,int(coordinate.x*w),int(coordinate.y*h))
            cv2.imshow('Mediapipe Feed', image)
            out.write(image)
        else:
            break
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()