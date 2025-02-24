from ultralytics import YOLO
import cv2
import numpy as np
import time
import firebase_admin
from firebase_admin import credentials, db

# Load the YOLO11 model
model = YOLO(r"y4_semester1\Yolo11\FirstVersion\best.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print ("Can not open webcam")
    exit()

# dẫn đến file json
cred = credentials.Certificate (r"y4_semester1\Raspberry_PI\Test\group7-android-firebase-adminsdk-uwp94-591b29394e.json")

# khởi tạo Firebase Admin SDK
firebase_admin.initialize_app(cred,{
    'databaseURL':'https://group7-android-default-rtdb.asia-southeast1.firebasedatabase.app/'
})

# tham chiếu đên status
ref = db.reference('user')

#condition for timer
startTime = None
#conditionMeet = False

#Start 
while True:
    #read frame cam 
    ret, frame = cap.read()
    if not ret:
        print("Can not read frame from webcam")
        break

    

    #show = False = dont need to draw box
    results = model.predict(frame, show = False)
    for result in results:
        # Access bounding box data
        boxes = result.boxes  # All bounding boxes
        if boxes is not None:
            labels = boxes.cls  # Class indices
            scores = boxes.conf  # Confidence scores

            # Loop through each label and print it
            for label, score in zip(labels, scores):
                label_name = result.names[int(label)]  # Convert class index to name
                print(f"Label: {label_name}, Confidence: {score:.2f}")
            
     # Truyền lên firebase
    ref.child('Cus').set({
        'State': f"{label_name} (Confident: {score:.2f})", 
        'Time' :'0:00',
        'Name' :'Nhan',
        'Age'  :'21'
    })

    # Alert if drowsy > 3.5s
    if label_name == "drowsy" and score > 0.5:
        if startTime is None:
            startTime = time.time()
        elif time.time() - startTime > 3.5:
            print("Dây mauuuuuu")
            #conditionMeet = True
    
    elif label_name == "awake":
        startTime = None
       # conditionMeet = False
    
    #draw box
    annotated_frame = results[0].plot()

    cv2.imshow("Detection", annotated_frame)

     # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()




  # # Convert the results to a format that OpenCV can display
    # annotated_frame = np.array(results.pandas().xyxy[0])  # Assuming you have a 'xyxy' format for bounding boxes

    # # if isinstance(results, list) and len(results) > 0:
    # #     boxs = results[0].xyxy[0].cpu().numpy()  # Chuyển về mảng NumPy
    # frame_with_boxes = cv2.rectangle(frame.copy(), 
    #                                 (int(annotated_frame['xmin']), int(annotated_frame['ymin'])), 
    #                                 (int(annotated_frame['xmax']), int(annotated_frame['ymax'])), 
    #                                 (0, 255, 0), 2)  # Green bounding box

    # Convert the results to a format that OpenCV can display
    # annotated_frame = np.array(results.pandas().xyxy[0])  # Assuming you have a 'xyxy' format for bounding boxes
    # frame_with_boxes = cv2.rectangle(frame.copy(), 
    #                                 (int(annotated_frame['xmin']), int(annotated_frame['ymin'])), 
    #                                 (int(annotated_frame['xmax']), int(annotated_frame['ymax'])), 
    #                                 (0, 255, 0), 2)  # Green bounding box

    # if isinstance(results, list) and len(results) > 0:
    #     boxes = results[0].boxes.xyxy.cpu().numpy()  # Chuyển về mảng NumPy
    #     for (x1, y1, x2, y2) in boxes:
    #         frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Vẽ hộp giới hạn
