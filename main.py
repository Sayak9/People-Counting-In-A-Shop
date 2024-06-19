import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*


model=YOLO('yolov8m.pt')

area1=[((251,157), (191,197),(306,351), (360,289))]

area2=[(185,205),(161,239),(276,385), (296,355)]



def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('demo.mp4')

tracker = Tracker()
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)
people_entering = {}
people_exiting = {}
count=0
entering = set()
exiting = set()


fourcc = cv2.VideoWriter_fourcc(*'XVID')  
fps = cap.get(cv2.CAP_PROP_FPS)  
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (852, 480))

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    frame=cv2.resize(frame,(852,480))
#    frame=cv2.flip(frame,1)
    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list=[]
             
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        
        if 'person' in c:
           list.append([x1,y1,x2,y2])
    bbox_id = tracker.update(list)
    
    for bbox in bbox_id:
        x3,y3,x4,y4,id = bbox
        
        print(people_entering)
        print(people_exiting)
    
        results = cv2.pointPolygonTest(np.array(area2,np.int32), ((x4,y4)), False)
        if results>=0:
               print("Person detected in area 2 exiting")
               people_exiting[id]=(x4,y4)
               
        if id in people_exiting:
            results_exiting = cv2.pointPolygonTest(np.array(area1,np.int32), ((x4,y4)), False)
            
            if results_exiting>=0:
               print("Person detected in area 1 exiting")
               exiting.add(id)
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
               cv2.circle(frame, (x4,y4),4,(255,0,255),-1) 
               cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
               
               
                
               
        results2 = cv2.pointPolygonTest(np.array(area1,np.int32), ((x4,y4)), False)
        if results2>=0:
               print("Person detected in area 1 entering")
               people_entering[id]=(x4,y4)
               
        if id in people_entering:
            results_entering = cv2.pointPolygonTest(np.array(area2,np.int32), ((x4,y4)), False)
            
            if results_entering>=0:
               print("Person detected in area 2 entering")
               entering.add(id) 
               cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
               cv2.circle(frame, (x4,y4),4,(255,0,255),-1) 
               cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
               #exiting.add(id)       
             
        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
   # cv2.putText(frame,str('1'),(504,471),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,255),2)
   # cv2.putText(frame,str('2'),(466,485),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,0),1)
    
    i = (len(entering))
    j = (len(exiting))
    
    cv2.putText(frame,str(i),(66,85),cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,0,255),2)
    cv2.putText(frame,str(j),(66,145),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,0,0),2)
    

    cv2.imshow("RGB", frame)
    out.write(frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

