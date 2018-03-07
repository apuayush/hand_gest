import cv2
import numpy as np

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    c = 1
    while True:
        ret_value, frame = cam.read()
        # print(ret_value)

        if ret_value == True:
            print("saving")
            cv2.imwrite('/home/apurvnit/Projects/hand_gest/yolo_trainer/images/attemp1_'+str(c)+'.jpg', frame)
            c+=1
        cv2.imshow("images",frame)


        keypress = cv2.waitKey(1) & 0xFF

        # if the user pressed "q", then stop looping
        if keypress == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()