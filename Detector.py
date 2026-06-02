import cv2
from time import time
from PIL import Image
from tkinter import messagebox
from camera_utils import create_video_capture, read_frame


def safe_showerror(title, msg):
    try:
        messagebox.showerror(title, msg)
    except Exception:
        print(f"ERROR: {title}: {msg}")


def safe_showinfo(title, msg):
    try:
        messagebox.showinfo(title, msg)
    except Exception:
        print(f"INFO: {title}: {msg}")


def main_app(name, timeout = 5):
        
        face_cascade = cv2.CascadeClassifier('./data/haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(f"./data/classifiers/{name}_classifier.xml")
        cap = create_video_capture(0)
        if not cap.isOpened():
            safe_showerror('Camera Error', 'Could not open webcam or fallback video.')
            return
        pred = False
        start_time = time()
        while True:
            ret, frame = read_frame(cap)
            if not ret or frame is None:
                messagebox.showerror('Camera Error', 'Failed to read from webcam or fallback video.')
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:


                roi_gray = gray[y:y+h,x:x+w]

                id,confidence = recognizer.predict(roi_gray)
                confidence = 100 - int(confidence)
                if confidence > 50:
                    #if u want to print confidence level
                            #confidence = 100 - int(confidence)
                        pred = True
                        text = 'Recognized: '+ name.upper()
                        font = cv2.FONT_HERSHEY_PLAIN
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 255, 0), 1, cv2.LINE_AA)
                       
                       
                else:   
                        pred = False
                        text = "Unknown Face"
                        font = cv2.FONT_HERSHEY_PLAIN
                        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        frame = cv2.putText(frame, text, (x, y-4), font, 1, (0, 0,255), 1, cv2.LINE_AA)
                       
                        
            cv2.imshow("image", frame)

            '''
            if cv2.waitKey(20) & 0xFF == ord('q'):
                print(pred)
                if pred == True :
                    
                    dim =(124,124)
                    img = cv2.imread(f".\\data\\{name}\\{pred}{name}.jpg", cv2.IMREAD_UNCHANGED)
                    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
                    cv2.imwrite(f".\\data\\{name}\\50{name}.jpg", resized)
                    Image1 = Image.open(f".\\2.png") 
                      
                    # make a copy the image so that the  
                    # original image does not get affected 
                    Image1copy = Image1.copy() 
                    Image2 = Image.open(f".\\data\\{name}\\50{name}.jpg") 
                    Image2copy = Image2.copy() 
                      
                    # paste image giving dimensions 
                    Image1copy.paste(Image2copy, (195, 114)) 
                      
                    # save the image  
                    Image1copy.save("end.png") 
                    frame = cv2.imread("end.png", 1)
                    cv2.imshow("Result",frame)
                    cv2.waitKey(5000)
                
                    messagebox.showinfo('Congrat', 'You have already checked in')
                else:
                    messagebox.showerror('Alert', 'Please check in again')
                break
        '''
            elapsed_time = time() - start_time
            if elapsed_time >= timeout:
                print(pred)
                if pred:
                    safe_showinfo('Congrat', 'You have already checked in')
                else:
                    safe_showerror('Alert', 'Please check in again')
                break

            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
        
        
