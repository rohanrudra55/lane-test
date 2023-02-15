import cv2 
import numpy as np 
import matplotlib.pyplot as plt

from utils import wrap, fit_polynomial

cap = cv2.VideoCapture("solidYellowLeft.mp4")
ret, frame = cap.read() 
print(frame.shape)




while True:
    ret, frame = cap.read() 
    bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(bw,140,255,cv2.THRESH_BINARY)

    warped = wrap(thresh1)/255.
    out_img = fit_polynomial(warped)

    # cv2.imshow("wrap", warped)
    # cv2.imshow("out", out_img)


    # Press 'Esc' to Exit

    if cv2.waitKey(25) == 27:
        break 

cap.release()
cv2.destroyAllWindows()

