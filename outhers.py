import  numpy as np
import cv2
import sys
from random import randint



#escolher o filtro que será aplicado a imagem, operaçoes morfologicasdiddila
def get_filter(img,filter):
    kernel_elipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))#dilation e erosion
    
    kernel_ret = np.ones((3,3), np.uint8)#opening e closing
    op = None        

    if filter == 'closing':
        op =  cv2.morphologyEx(img,cv2.MORPH_ClOSE,kernel_ret,iterations=2)
        
    elif filter == 'opening': 
        op =  cv2.morphologyEx(img,cv2.MORPH_OPEN, kernel_ret ,iterations=2)
        
    elif filter == 'dilation':
        op =  cv2.dilate(img,kernel_elipse   ,iterations=2)
        
    elif filter == 'erosion':
        op =  cv2.erode(img, kernel_elipse, iterations=2)
        
    elif filter == 'combine':
        closing = cv2.morphologyEx(img,cv2.MORPH_ClOSE,kernel_ret,np.uint8  ,iterations=2)
        opening =  cv2.morphologyEx(closing,cv2.MORPH_OPEN, kernel_ret,np.uint8  ,iterations=2)
        op =  cv2.morphologyEx(opening,kernel_elipse  ,iterations=2)
        
    return op

def getSubttractor(model):
    
    if model == "GMG":
        return cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames = 120, decisionThreshold = 0.8)
    if model == "MOG":
        return cv2.bgsegm.createBackgroundSubtractorMOG(history = 200, nmixtures = 5, backgroundRatio = 0.7, noiseSigma = 0)
    if model == "MOG2":
        return cv2.createBackgroundSubtractorMOG2(history = 500, detectShadows = True, varThreshold=100)
    if model == "KNN":
        return cv2.createBackgroundSubtractorKNN(history = 500, dist2Threshold= 400, detectShadows = True)
    if model =="CNT":
        return cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, useHistory = True, maxPixelStability=15*60, isParallel = True)


text_color = (randint(0,255),randint(0,255),randint(0,255))
order_color = (randint(0,255),randint(0,255),randint(0,255))
font = cv2.FONT_HERSHEY_SIMPLEX
video = "videos/Traffic_4.mp4"

print(text_color)

BGS = ['GMG','MOG2','MOG','KNN','CNT']

cap = cv2.VideoCapture(video)
model = getSubttractor(BGS[1])
print(cap)
def main():
    
    while cap.isOpened():
        
        ok, frame = cap.read()
        
        frame = cv2.resize(frame, (0,0), fx = 0.5, fy = 0.5)
        
        
        mask = model.apply(frame)
        filt_mask = get_filter(mask,"dilation")
        res = cv2.bitwise_and(frame, frame, mask=mask)
        
        if not ok:
            print("End of the video")
            break
        
        cv2.imshow("dilation",filt_mask)
        cv2.imshow("mask",mask)
        cv2.imshow("lugarres",res)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
main()    