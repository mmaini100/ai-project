import cv2 #importing library


# Loading Haar Cascade Classifiers
haarcascade= cv2.data.haarcascades + "haarcascade_frontalface_default.xml" 
harcascade = cv2.data.haarcascades + "haarcascade_eye.xml"
eye_cascade = cv2.CascadeClassifier(harcascade)
face_cascade = cv2.CascadeClassifier(haarcascade)


#Accessing the Camera
video_cap = cv2.VideoCapture(0)

#Main Loop for Real-time Video Processing
while True :
    _ , img = video_cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Converting Image to Grayscale
    face = face_cascade.detectMultiScale(gray,1.1,4)
    for(x,y,w,h) in face:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2) #Drawing Rectangles Around Faces
        roi_gray = gray[y:y+h, x:x+w] # Region of Interest (ROI) for Eyes
        roi_color = img[y:y+h, x:x+w]
    eye = eye_cascade.detectMultiScale(roi_gray) #Detecting Eyes
    for(ex,ey,ew,eh) in eye:
             cv2.rectangle(roi_color, (ex,ey), (ex+ew,ey+eh), (0,255,0),2) #Drawing Rectangles Around Eyes
              
    cv2.imshow("video_live", img) #Displaying the video


   #Exit Condition
    if cv2.waitKey(1) == ord("a"): #camera stops by typing 'a'
        break
        video_cap.release()
        cv2.destroyAllWindows()
 
