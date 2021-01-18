#!/usr/bin/env python
# coding: utf-8

#importing libraries
import cv2
import sys
import numpy as np
import copy
import math


try:
    def getCountOfFingers():
        #parameters
        cap_region_x_begin=0.5  # start point of the region - x axis
        cap_region_y_end=0.8  # start point of the region - y axis
        threshold = 60  # threshold value
        blurValue = 41  # GaussianBlur value
        bgSubThreshold = 50 #backgroundThreshold value
        learningRate = 0

        # variables
        isBgCaptured = 0   # boolean, whether the background captured
        triggerSwitch = False  # if true keyborad simulate works

    
        def printThreshold(thr):
            print("! Changed threshold to "+str(thr))

        try:
            def removeBackground(frame):
                fgmask = bgModel.apply(frame,learningRate=learningRate) #foreground mask
                kernel = np.ones((3, 3), np.uint8) #creating a numpy array of ones
                fgmask = cv2.erode(fgmask, kernel, iterations=1) #cv2.erode for removing noise
                res = cv2.bitwise_and(frame, frame, mask=fgmask) #cv2.bitwsie_and method using frame - source 1 and frame as destimation, mask = fgmask
                return res
        except ValueError:
            print("Error while removng bakcground! ",sys.exc_info()[0],"occured.")
            


        try:
            def countFingers(res,drawing):
                #  convexity defect
                hull = cv2.convexHull(res, returnPoints=False) #finds convexhull of a set point, points = res(2d point set), return point is false
                if len(hull) > 3:
                    defects = cv2.convexityDefects(res, hull) #function finds all convexity defects of the input contour and returns a sequence of the CvConvexityDefect structures
                    if type(defects) != type(None):  #if defect is not null, do the below
                        cnt = 0
                        for i in range(defects.shape[0]):  # calculate the angle
                            s, e, f, d = defects[i][0]
                            start = tuple(res[s][0])
                            end = tuple(res[e][0])
                            far = tuple(res[f][0])
                            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) #finding the distance between two points
                            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
                            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
                            angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))  # thorem cosine 
                            if angle <= math.pi / 2:  # angle less than 90 degree, treat as fingers
                                cnt += 1
                                cv2.circle(drawing, far, 8, [211, 84, 0], -1) #drawing circle on numpy array of fixed radius and lenth
                        return True, cnt
                return False, 0
        except ValueError:
            print("Error while removng bakcground! ",sys.exc_info()[0],"occured.")
            
        # Camera part
        camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        camera.set(10,200) #setting frame of the camera
        cv2.namedWindow('trackbar') #assigning name to the camera window
        cv2.createTrackbar('trh1', 'trackbar', threshold, 100, printThreshold) 

        while camera.isOpened():
            ret, frame = camera.read() #activaes the camera for image capturing
            threshold = cv2.getTrackbarPos('trh1', 'trackbar')
            frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothening filter noise removal while preserving edges. But the operation is slower 
            frame = cv2.flip(frame, 1)  # filip the frame horizontally
            cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                             (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2) #draw rectangle on any image
            cv2.imshow('original', frame) #shows the captured result

         #  Main operation
            if isBgCaptured == 1:  # this part wont run until background captured
                img = removeBackground(frame) #once background is captured, removal of noise is done
                img = img[0:int(cap_region_y_end * frame.shape[0]),
                            int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]  # clip the ROI


                    # convert the image into binary image
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #converting color of the image
                blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0) #blurring hte image arund the edges
                ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

                # get the contours
                thresh1 = copy.deepcopy(thresh) #deep copy creates a new object and recursively adds the copies of nested objects present in the original elements.
                contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #detect objects in an image, RETR_TREE- retrieves all the contours and creates a full family hierarchy, CHAIN_ARRPOX_SIMPLE - It removes all redundant points and compresses the contour, thereby saving memory
                length = len(contours) #getting lenght of contors are we want only the count
                maxArea = -1
                if length > 0:
                    for i in range(length):  # find the biggest contour (according to area)
                        temp = contours[i]
                        area = cv2.contourArea(temp)
                        if area > maxArea:
                            maxArea = area
                            ci = i
                            
                    res = contours[ci]
                    hull = cv2.convexHull(res)
                    drawing = np.zeros(img.shape, np.uint8)
                    cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)#drawingg the contors on res
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)#drawing the hull
                    
                    isFinishCal,cnt = countFingers(res,drawing) #passing the values to find res, drawing
                    if triggerSwitch is True:
                        if isFinishCal is True and cnt >= 1:
                            print ("no of fingers",cnt+1)
                            camera.release()
                            cv2.destroyAllWindows()
                            return cnt+1
                            break
                        elif cnt==0:
                            print ("no of fingers",1)
                            camera.release()
                            cv2.destroyAllWindows()
                            return 1
                            break

                cv2.imshow('output', drawing)

                 
            k = cv2.waitKey(10)
            if k == 27:  # press ESC to exit
                camera.release()
                cv2.destroyAllWindows()
                break
            elif k == ord('b'):  # press 'b' for background capture
                bgModel = cv2.createBackgroundSubtractorMOG2(0, bgSubThreshold)
                isBgCaptured = 1
            elif k == ord('r'):  # press 'r' for reset background
                bgModel = None #intializing the backgroundand other variables
                triggerSwitch = False
                isBgCaptured = 0
            elif k == ord('n'):
                triggerSwitch = True

except:
    print("Oops!",sys.exc_info()[0],"occured.")
    camera.release()
    cv2.destroyAllWindows()
finally:
    cv2.destroyAllWindows()

