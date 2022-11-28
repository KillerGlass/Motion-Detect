import  numpy as np
import cv2
import csv
from random import randint

fp = open('reporter.csv', 'w')
writer = csv.DictWriter(fp, fieldnames=["Frame", "Pixel Count"])
writer.writeheader()

text_color = (randint(0,255),randint(0,255),randint(0,255))
order_color = (randint(0,255),randint(0,255),randint(0,255))
font = cv2.FONT_HERSHEY_SIMPLEX
test_size = 1.2
video = "videos/people.mp4"
title_pos = (100,40)
BGS = ['GMG','MOG2','MOG','KNN','CNT']

def getSubttractor(model):
    
    if model == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG()
    if model == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG()
    if model == "MOG2":
        return cv2.createBackgroundSubtractorMOG2()
    if model == "KNN":
        return cv2.createBackgroundSubtractorKNN()
    if model =="CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT()

    
list_model = []
for i in BGS:
    list_model.append(getSubttractor(i))
    
cap = cv2.VideoCapture(video)
    
framecount = 0
def main():

    while cap.isOpened():
        
        ok, frame = cap.read()
        
    
        if not ok:
            print("End of the video")
            break
        
    
        frame = cv2.resize(frame, (0,0), fx = 0.2, fy = 0.2)
        #framecount  +=1

        gmg = list_model[0].apply(frame)
        mog = list_model[1].apply(frame)
        mog2 = list_model[2].apply(frame)
        knn = list_model[3].apply(frame)
        cnt = list_model[4].apply(frame)
        
        gmgCount = np.count_nonzero(gmg)
        mogCount = np.count_nonzero(mog)
        mog2Count = np.count_nonzero(mog2)
        knnCount = np.count_nonzero(knn)
        cntCount = np.count_nonzero(cnt)
        
        writer.writerow({'Frame': 'Mog', "Pixel Count": mogCount})
        writer.writerow({'Frame': 'Mog2', "Pixel Count": mog2Count})
        writer.writerow({'Frame': 'gmg', "Pixel Count": gmgCount})
        writer.writerow({'Frame': 'knn', "Pixel Count": knnCount})
        writer.writerow({'Frame': 'cnt', "Pixel Count": cntCount})
        
        k = cv2.waitKey(0) & 0xFF 
        
        cv2.imshow("Mog",mog)
        cv2.imshow("Mog2",mog2)
        cv2.imshow("knn",knn)
        cv2.imshow("cnt",cnt)
        cv2.imshow("gmg",gmg)
        cv2.imshow("frame",frame)
        
        cv2.moveWindow('frame',0,0)
        cv2.moveWindow('Mog',0,250)
        cv2.moveWindow('Mog2',0,500)
        cv2.moveWindow('knn',719,0)
        cv2.moveWindow('cnt',719,250)
        cv2.moveWindow('gmg',719,500)

        
        if k == 27:
            break
            
main()
        
