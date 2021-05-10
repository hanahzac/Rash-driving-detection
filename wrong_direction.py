import cv2
import numpy as np
import vehicles
import time

cnt_up=0
cnt_down=0
cnt_wrong_up = 0
cnt_wrong_down = 0

#cap=cv2.VideoCapture("NumberplatesDetector\Freewa.mp4")
#cap=cv2.VideoCapture("captures\left_3.mp4")
cap=cv2.VideoCapture("captures\capture_3.mp4")

#Get width and height of video

w=cap.get(3)
h=cap.get(4)
frameArea=h*w
areaTH=frameArea/400

#Lines
line_up=int(3.5*(h/5))
line_down=int(4*(h/5))

up_limit=int(3*(h/5))
down_limit=int(4.5*(h/5))

print("Red line y:",str(line_down))
print("Blue line y:",str(line_up))
line_down_color=(255, 0, 0)
line_up_color=(255, 0, 255)
pt1 =  [0, line_down]
pt2 =  [w, line_down]
pts_L1 = np.array([pt1,pt2], np.int32)
pts_L1 = pts_L1.reshape((-1,1,2))
pt3 =  [0, line_up]
pt4 =  [w, line_up]
pts_L2 = np.array([pt3,pt4], np.int32)
pts_L2 = pts_L2.reshape((-1,1,2))

pt5 = [0, up_limit]
pt6 =  [w, up_limit]
pts_L3 = np.array([pt5,pt6], np.int32)
pts_L3 = pts_L3.reshape((-1,1,2))
pt7 =  [0, down_limit]
pt8 =  [w, down_limit]
pts_L4 = np.array([pt7,pt8], np.int32)
pts_L4 = pts_L4.reshape((-1,1,2))

#Background Subtractor
fgbg=cv2.createBackgroundSubtractorMOG2(detectShadows=True)

#Kernals
kernalOp = np.ones((3,3),np.uint8)
kernalOp2 = np.ones((5,5),np.uint8)
kernalCl = np.ones((11,11),np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX
cars = []
max_p_age = 5
pid = 1


while(cap.isOpened()):
    ret,frame=cap.read()
    for i in cars:
        i.age_one()
    fgmask=fgbg.apply(frame)
    fgmask2=fgbg.apply(frame)

    if ret==True:

        #Binarization
        ret,imBin=cv2.threshold(fgmask,200,255,cv2.THRESH_BINARY)
        ret,imBin2=cv2.threshold(fgmask2,200,255,cv2.THRESH_BINARY)
        #OPening i.e First Erode the dilate
        mask=cv2.morphologyEx(imBin,cv2.MORPH_OPEN,kernalOp)
        mask2=cv2.morphologyEx(imBin2,cv2.MORPH_CLOSE,kernalOp)

        #Closing i.e First Dilate then Erode
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernalCl)
        mask2=cv2.morphologyEx(mask2,cv2.MORPH_CLOSE,kernalCl)

        # road sides masking - red & blue
        height, width, channels = frame.shape
        polygon1 = np.array([
            [(0, 0), ((int)(width/1.6), 0), 
            ((int)(width/5), height), (0, height)]
            ])
        mask1 = np.zeros_like(frame)
        polygon2 = np.array([
            [((int)(width/1.6), 0), (width, 0), 
            (width, height), ((int)(width/5), height)]
            ])
        mask2 = np.zeros_like(frame)
        '''polygon1 = np.array([
            [(0, 0), ((int)(width/2), 0), 
            ((int)(width/2), height), (0, height)]
            ])
        mask1 = np.zeros_like(frame)
        polygon2 = np.array([
            [((int)(width/2), 0), (width, 0), 
            (width, height), ((int)(width/2), height)]
            ])
        mask2 = np.zeros_like(frame)'''

        #Find Contours
        countours0, hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        for cnt in countours0:
            area=cv2.contourArea(cnt)
            # print(area)
            if area>areaTH:
                ####Tracking######
                m=cv2.moments(cnt)
                cx=int(m['m10']/m['m00'])
                cy=int(m['m01']/m['m00'])
                x,y,w,h=cv2.boundingRect(cnt)

                new=True
                if cy in range(up_limit,down_limit):
                    for i in cars:
                        if abs(x - i.getX()) <= w and abs(y - i.getY()) <= h:
                            new = False
                            i.updateCoords(cx, cy)

                            if i.line_side(i.getX(), i.getY(), width, height) > 0:
                                # > 0 is left side of line
                                if i.going_DOWN(line_down,line_up)==True:
                                    cnt_up += 1
                                    print("ID:",i.getId(),'crossed going up at', time.strftime("%c"))
                                elif i.going_UP(line_down, line_up)==True:
                                    cnt_wrong_down += 1
                                    print("ID:", i.getId(), 'wrongly going up at', time.strftime("%c"))
                            else:
                                # right side of line
                                if i.going_UP(line_down,line_up)==True:
                                    cnt_down += 1
                                    print("ID:",i.getId(),'crossed going down at', time.strftime("%c"))
                                elif i.going_DOWN(line_down, line_up)==True:
                                    cnt_wrong_up += 1
                                    print("ID:", i.getId(), 'wrongly going down at', time.strftime("%c"))
                            break
                        if i.getState()=='1':
                            if i.getDir()=='down'and i.getY()>down_limit:
                                i.setDone()
                            elif i.getDir()=='up'and i.getY()<up_limit:
                                i.setDone()
                        if i.timedOut():
                            index=cars.index(i)
                            cars.pop(index)
                            del i

                    if new==True: #If nothing is detected,create new
                        p=vehicles.Car(pid,cx,cy,max_p_age)
                        cars.append(p)
                        pid+=1

                cv2.circle(frame,(cx,cy),5,(0,0,255),-1)
                img=cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

        for i in cars:
            cv2.putText(frame, str(i.getId()), (i.getX(), i.getY()), font, 0.3, (255, 255, 255), 1, cv2.LINE_AA)

        str_up='CORRECT: '+str(cnt_up+cnt_down)
        str_down='WRONG DIRECTION: '+str(cnt_wrong_up+cnt_wrong_down)
        frame=cv2.polylines(frame,[pts_L1],False,line_down_color,thickness=2)
        frame=cv2.polylines(frame,[pts_L2],False,line_up_color,thickness=2)
        frame=cv2.polylines(frame,[pts_L3],False,(255,255,255),thickness=1)
        frame=cv2.polylines(frame,[pts_L4],False,(255,255,255),thickness=1)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_up, (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, str_down, (10, 90), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        # Fill poly-function deals with multiple polygon
        cv2.fillPoly(mask1, polygon1, (255, 0, 0))
        cv2.fillPoly(mask2, polygon2, (0, 0, 255))

        notmask1 = cv2.bitwise_not(mask1)
        second_videopart = cv2.bitwise_and(frame, mask2)
        first_videopart = cv2.bitwise_and(frame, mask1)

        maskedFrame = cv2.bitwise_or(first_videopart, second_videopart)

        cv2.imshow('Input Stream',maskedFrame)

        if cv2.waitKey(1)&0xff==ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()










