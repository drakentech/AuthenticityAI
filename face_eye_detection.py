#!/usr/bin/python3
import numpy as np
import cv2
import math
from statistics import variance
import itertools
import time

face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_alt.xml')
eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye.xml')
#eye_cascade = cv2.CascadeClassifier('models/haarcascade_eye_tree_eyeglasses.xml')
mouth_cascade = cv2.CascadeClassifier('models/haarcascade_mcs_mouth.xml')
#eye_cascade = cv2.CascadeClassifier('haarcascade_mcs_lefteye.xml')

scale_factor = 1.2
font = cv2.FONT_ITALIC
font_color = (255, 255, 255)
face_min_neighbors = 3
face_min_size = (100, 100)
face_max_size = (2000, 2000)

eye_min_neighbors = 3
eye_min_size = (50, 50)
eye_max_size = (100, 100)

mouth_min_neighbors = 2
mouth_min_size = (50, 50)
mouth_max_size = (200, 200)

angleslist = []
goodtriangleslist = []
angles = 0
goodblurlist = []
blurlist = []

# Plot
from matplotlib import pyplot as plt 

#webcam=True #if working with video file then make it 'False'
webcam=True

def is_blurred(image, threshold=100):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the variance of Laplacian
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    # Check if the variance is below the threshold
    return variance < threshold

def calculate_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def calculate_angles(points):
    angles = []
    for i in range(len(points)):
        p1, p2, p3 = points[i], points[(i + 1) % len(points)], points[(i + 2) % len(points)]

        # Calculate vectors
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Calculate dot product and magnitudes
        dot_product = v1[0] * v2[0] + v1[1] * v2[1]
        magnitude_v1 = math.sqrt(v1[0]**2 + v1[1]**2)
        magnitude_v2 = math.sqrt(v2[0]**2 + v2[1]**2)
        angle_radians = math.acos(min(max(dot_product / (magnitude_v1 * magnitude_v2), -1.0), 1.0))

        angle_degrees = math.degrees(angle_radians)
        angles.append(angle_degrees)
    return angles

def detect():
    if webcam:
        video_cap = cv2.VideoCapture(0) # use 0,1,2..depanding on your webcam
    else:
        video_cap = cv2.VideoCapture("videocall_blur.webm")
    while True:
        # Capture frame-by-frame
        ret, img = video_cap.read()

        #converting to gray image for faster video processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=face_min_neighbors, minSize=face_min_size, maxSize=face_max_size)

        # if at least 1 face detected
        if len(rects) >= 0:
            artifactlist=[]
            artifactlist_mouth = []
            # Draw a rectangle around the faces
            for (x, y, w, h) in rects:
                # Crop image for blur check
                crop_img = img[y:y+h, x:x+w]
                #cv2.namedWindow("Display1", cv2.WINDOW_AUTOSIZE)
                #cv2.imshow('Display1', crop_img)
                if is_blurred(crop_img) == True:
                    goodblurlist.append(is_blurred(crop_img))
                else:
                    blurlist.append(is_blurred(crop_img))
                time.sleep(0.1)
                color = (0, 0, 255)
                fcenter = (int(x+(w/2)), int(y+(h/2)))
                rectangle_text = cv2.putText(img, str(fcenter), fcenter, font, 0.5, font_color)
                rectangle_circle = cv2.circle(rectangle_text, fcenter , 5, color, -1)
                rectangle_img = cv2.rectangle(rectangle_circle, (x, y), (x + w, y + h), (0, 255, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = rectangle_img[y:y+h, x:x+w]
                artifactlist.append(fcenter)

                # TODO: compare histogram
                #histogram = cv2.calcHist([roi_color], [0], None, [256], [0, 256])
                #print(histogram)
                #plt.plot(histogram)
                #plt.show()
                
                eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=eye_min_neighbors, minSize=eye_min_size, maxSize=eye_max_size)
                for (ex,ey,ew,eh) in eyes:
                    color = (0, 0, 255)
                    ecenter = (int(ex+(ew/2)), int(ey+(eh/2)))
                    if ecenter[1] > int(h/2):
                        print("Not valid eye coord: ", ecenter)
                    else:
                        print("Eye center coordinate : ", ecenter)

                        roi_color = cv2.circle(roi_color, ecenter , 5, color, -1)
                        roi_color = cv2.putText(roi_color,str(ecenter), ecenter, font, 0.5, font_color)
                        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(213,255,0),2)
                        artifactlist.append(ecenter)

                    mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=scale_factor, minNeighbors=mouth_min_neighbors, minSize=mouth_min_size, maxSize=mouth_max_size)

                    for (mx, my, mw, mh) in mouth:
                        color = (0, 100, 244)
                        mcenter = (int(mx+(mw/2)), int(my+(mh/2)))
                        if mcenter[1] < int(h/2):
                            pass
                        else:
                            roi_color = cv2.rectangle(roi_color, (mx,my), (mx+mw, my+mh), color, 1)
                            cv2.circle(roi_color, mcenter, 5, color, -1)
                            roi_color = cv2.putText(roi_color,str(mcenter), mcenter, font, 0.5, font_color)
                            cv2.line(roi_color, (int(mx+(mw/2)), int(my+(mh/2))), (int(ex+(ew/2)), int(ey+(eh/2))) ,(59, 0, 225), 1)
                            artifactlist_mouth.append(mcenter)

            # Display the resulting frame
            distances = {}
            for i, j in itertools.combinations(range(len(artifactlist)), 2):
                distance = calculate_distance(artifactlist[i], artifactlist[j])
                distances[(i, j)] = distance

            try:
                if len(artifactlist)>0:
                    angles = calculate_angles(artifactlist)
                    '''for i, angle in enumerate(angles, 1):
                        print(f"Angle {i}: {angle:.2f} degrees")'''

                    if int(sum(angles)) == 180:
                        goodtriangleslist.append(angles)

                    angleslist.append(int(sum(angles)))
                else:
                    print("Artifactlist is empty!")
            except Exception as e:
                print(f"An error has occoured: {e}")


            cv2.imshow('Face Detection on Video', img)
            #wait for 'c' to close the application
            if cv2.waitKey(1) & 0xFF == ord('c'):
                break
    video_cap.release()

def main():
    detect()
    cv2.destroyAllWindows()
    for sublist in goodtriangleslist:
        sublist.sort()
    transposed_data = list(zip(*goodtriangleslist))
    print("Blurred rate: ", "{:.3f}".format(len(goodblurlist)/len(blurlist)))
    v = []
    for i, numbers in enumerate(transposed_data, 1):
        var = variance(numbers)
        v.append(var)
        #print(f"Variance of element {i}: {var:.2f}")
    print("Variance median: ", "{:.3f}".format(np.median(v)))
    print("Success rate: ", "{:.3f}".format(len(goodtriangleslist)/len(angleslist)))

if __name__ == "__main__":
    main()
