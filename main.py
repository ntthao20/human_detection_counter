import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import cvzone
import numpy as np

model=YOLO('yolov8s.pt')



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        point = [x, y]
        print(point)
  
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)
cap=cv2.VideoCapture('p.mp4')


my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0

tracker = Tracker()

# Area 1 interestic area for detecting and tracking
area1=[(494,289),(505,499),(578,496),(530,292)]
# Area 2 direction in or out
area2=[(548,290),(600,496),(637,493),(574,288)]

direc_out = {}
direc_in = {}

counter_out = []
counter_in = []
while True:    
    ret,frame = cap.read()
    if not ret:
        break


#    count += 1
#    if count % 3 != 0:
#        continue
    frame=cv2.resize(frame,(1020,500))
   

    results=model.predict(frame)
#   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#   print(px)
    
    list = []
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        
        c=class_list[d]
        if 'person' in c:
            #Create a list of Rectangle Coordinates
            list.append([x1, y1, x2, y2])
    bbox_idx = tracker.update(list)
    for bbox in bbox_idx:
        x3, y3, x4, y4, id = bbox
        results = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
        if results >= 0:
            direc_out [id] = (x4, y4)
        if id in direc_out:
            results1 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
            if results1 >= 0:
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 255, 255), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                if counter_out.count(id) == 0:
                    counter_out.append(id)

        results2 = cv2.pointPolygonTest(np.array(area1, np.int32), (x4, y4), False)
        if results2 >= 0:
            direc_in[id] = (x4, y4)
        if id in direc_in:
            results3 = cv2.pointPolygonTest(np.array(area2, np.int32), (x4, y4), False)
            if results3>= 0:
                cv2.circle(frame, (x4, y4), 4, (255, 0, 255), -1)
                cv2.rectangle(frame, (x3, y3), (x4, y4), (255, 0, 0), 2)
                cvzone.putTextRect(frame, f'{id}', (x3, y3), 1, 1)
                if counter_in.count(id) == 0:
                    counter_in.append(id)
    out_c = (len(counter_out))
    in_c = (len(counter_in))
    cvzone.putTextRect(frame, f'PERSION_IN: {in_c}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'PERSION_OUT: {out_c}', (50, 160), 2, 2)


    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(area2, np.int32)], True, (0, 255, 0), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()

