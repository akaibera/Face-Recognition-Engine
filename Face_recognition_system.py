import sys
import os
import numpy as np
import cv2
from cv2 import __version__
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog
from PIL import Image,ImageTk
from Emotion_model import EMODEL
import tensorflow as tf 


EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']
tf.reset_default_graph()
network = EMODEL()
network.build_network()
cap = cv2.VideoCapture(0)

class FaceDetector(object):
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifier(xml_path)
    def detect(self,image, biggest_only=True):      
        is_color = len(image) == 3
        if is_color:
           image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image       
        scale_factor = 1.2    
        min_neighbors = 5
        min_size = (30, 30)   
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
            cv2.CASCADE_DO_ROUGH_SEARCH if biggest_only else \
            cv2.CASCADE_SCALE_IMAGE
        face_coord = self.classifier.detectMultiScale(image_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags)
        return face_coord
    
class Operations(object):    
        def resize(self,images, size=(100, 100)):            
            images_norm = []
            for image in images:
                is_color = len(image.shape) == 3
                if is_color:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if image.shape < size:
                    image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
                else:
                    image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC)
                images_norm.append(image_norm)

            return images_norm

        def normalize_intensity(self,images):
            images_norm = []
            for image in images:
                is_color = len(image.shape) == 3
                if is_color:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images_norm.append(cv2.equalizeHist(image))
            return images_norm
        
        def cut_face_rectangle(self,image, face_coord):
                images_rectangle = []
                for (x, y, w, h) in face_coord:
                    images_rectangle.append(image[y: y + h, x: x + w])
                return input
            
        def draw_face_rectangle(self,image, faces_coord):
                for (x, y, w, h) in faces_coord:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (206, 0, 209), 2)
                return input
            
        def cut_face_ellipse(self,image, face_coord):
                images_ellipse = []
                for (x, y, w, h) in face_coord:
                    center = (int(x + w / 2),int( y + h / 2))
                    axis_major = int(h / 2)
                    axis_minor = int(w / 2)
                    mask = np.zeros_like(image)
                    mask = cv2.ellipse(mask,
                                       center=(center),
                                       axes=(axis_major, axis_minor),
                                       angle=0,
                                       startAngle=0,
                                       endAngle=360,
                                       color=(0, 0, 230),
                                       thickness=-1)
                    image_ellipse = np.bitwise_and(image, mask)
                    images_ellipse.append(image_ellipse[y: y + h, x: x + w])           
                return images_ellipse
            
        def draw_face_ellipse(self,image, faces_coord):
               for (x, y, w, h) in faces_coord:
                        center = (int(x + w / 2), int(y + h / 2))
                        axis_major = int(h / 2)
                        axis_minor = int(w / 2)
                        cv2.ellipse(image,
                                    center=(center),
                                    axes=(axis_major, axis_minor),
                                    angle=0,
                                    startAngle=0,
                                    endAngle=360,
                                    color=(0,0, 230),
                                    thickness=2)
               return image

op=Operations()     
class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self, in_grayscale=False):     
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def show_frame(self, seconds, in_grayscale=False):   
        _, frame = self.video.read()
        if in_grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('SnapShot', frame)
        key_pressed = cv2.waitKey(seconds * 1000)
        return key_pressed & 0xFF
    
def get_images(frame, faces_coord, shape):
        if shape == "rectangle":
            faces_img = op.cut_face_rectangle(frame, faces_coord)
            frame = op.draw_face_rectangle(frame, faces_coord)
        elif shape == "ellipse":
            faces_img = op.cut_face_ellipse(frame, faces_coord)
            frame = op.draw_face_ellipse(frame, faces_coord)
        faces_img = op.normalize_intensity(faces_img)
        faces_img = op.resize(faces_img)
        return (frame, faces_img)
    
def add_person(people_folder, shape):
             if not os.path.exists(PEOPLE_FOLDER):
               os.makedirs(PEOPLE_FOLDER)
             person_name= (simpledialog.askstring("Name","Enter name"))                    
             while True:
                 try:
                    peo_age =int(simpledialog.askstring("Age","Enter Age"))
                    break
                 except:
                     messagebox.showinfo("Valueerror","Please Enter a integer age: ")       
             folder1 = people_folder + person_name
             folder2=folder1 +"."+ str(peo_age)
             if not os.path.exists(folder2):
                messagebox.showinfo("Lets take some picture"," Click OK when ready.")
                os.mkdir(folder2)
                video = VideoCamera()
                detector = FaceDetector('haarcascade_frontalface_default.xml')
                counter = 1
                cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)
                while counter < 21:
                    frame = video.get_frame()
                    face_coord = detector.detect(frame)
                    if len(face_coord):
                        frame, face_img = get_images(frame, face_coord, shape)
                        cv2.imwrite(folder2 + '/' + str(counter) +'.jpg',
                                        face_img[0])
                        print ('Images Saved:'+ str(counter))
                        counter += 1
                        cv2.imshow('Saved Face: ', face_img[0])
                    cv2.imshow('Video Feed', frame)  
                    cv2.waitKey(50)
                cv2.destroyAllWindows()    
                   
             else:
                messagebox.showinfo("Title", "This name already exists.")      
        
detector = FaceDetector('haarcascade_frontalface_default.xml')        
def Eigen(people_folder, shape):
        try:
            people = [person for person in os.listdir(people_folder)]
        except:
            messagebox.showinfo("Train Error","Please add at least one person to the system")
        print ("This are the people in the Recognition System:")
        for person in people:
            print ("-" + person)
        recognizer1 = cv2.face.createEigenFaceRecognizer()
        recognizer2 = cv2.face.createEigenFaceRecognizer()
        threshold = 4000
        
        images = []
        labels1 = []
        labels2 = []
        labels_people = {}
        labels_age={}
        path=people_folder
        for i, person in enumerate(people):
            p_age= os.path.split(path+person)[-1].split('.')[1]
            p=person
            person=os.path.split(path+person)[-1].split('.')[0]
            labels_people[i] = person
            labels_age[i] = p_age
            for image in os.listdir(people_folder + p):           
                images.append(cv2.imread(people_folder + p + '/' + image, 0))
                labels1.append(i) 
                labels2.append(i)  
                                      
        try:
            recognizer1.train(images, np.array(labels1))
            recognizer2.train(images, np.array(labels2))
            if not os.path.exists('recognizer1Eigen'):
              os.mkdir('recognizer1Eigen')
            if not os.path.exists('recognizer2Eigen'):  
                os.mkdir('recognizer2Eigen')
            recognizer1.save('recognizer1Eigen/trainingData.yml')
            recognizer2.save('recognizer2Eigen/trainingData.yml')
        except:
            messagebox.showinfo("OpenCV Error", "Do you have at least one people in the database?")
            app.destroy()
            sys.exit()
        video = VideoCamera()
        while True:
            frame = video.get_frame()
            faces_coord = detector.detect(frame, False)
            if len(faces_coord):
                frame, faces_img = get_images(frame, faces_coord, shape)
                for i, face_img in enumerate(faces_img):
                    if __version__ == "3.1.0":
                        collector = cv2.face.MinDistancePredictCollector()
                        recognizer1.predict(face_img, collector)
                        recognizer2.predict(face_img, collector)
                        conf = collector.getDist()
                        pred = collector.getLabel()                    
                    else:
                        pred, conf = recognizer1.predict(face_img)
                        pred, conf = recognizer2.predict(face_img)
                    print ("Prediction: " + str(pred))
                    print ('Confidence: ' + str(round(conf)))
                    print ('Threshold: ' + str(threshold))
                    if conf < threshold:
                        cv2.putText(frame, "Name: "+labels_people[pred].capitalize(),
                                                    (faces_coord[i][0], faces_coord[i][1] - 20),
                                                    cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                                    cv2.LINE_AA)
                        cv2.putText(frame, "Age: "+labels_age[pred].capitalize(),
                                                    (faces_coord[i][0], faces_coord[i][1] + 2),
                                                    cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                                    cv2.LINE_AA)
                        for face in faces_coord:
                            (x,y,w,h)= face
                            newimg=frame[y:y+h,x:x+w]
                            newimg=cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
                            result=network.predict(newimg)
                            if result is not None:
                                result[0][2]-=0.15
                                result[0][4]-=0.15
                                if result[0][3] >0.06:
                                    result[0][3]+=0.4
                                maxindex=np.argmax(result[0]) 
                                cv2.putText(frame, EMOTIONS[maxindex],(faces_coord[i][0],
                                                  faces_coord[i][1] + 80),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                        cv2.LINE_AA)
                            cv2.imshow('Video', cv2.resize(frame,None,fx=1,fy=1,interpolation = cv2.INTER_CUBIC))
                                                    
                    else:
                        
                        cv2.putText(frame, "Unknown",
                                    (faces_coord[i][0], faces_coord[i][1]),
                                                        cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                                        cv2.LINE_AA)
                        for face in faces_coord:
                            (x,y,w,h)= face    
                            newimg=frame[y:y+h,x:x+w]
                            newimg=cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
                            result=network.predict(newimg)
                            if result is not None:
                                result[0][2]-=0.15
                                result[0][4]-=0.15
                                if result[0][3] >0.06:
                                    result[0][3]+=0.4
                                maxindex=np.argmax(result[0]) 
                                cv2.putText(frame, EMOTIONS[maxindex],(faces_coord[i][0],
                                                  faces_coord[i][1] + 80),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                        cv2.LINE_AA)
                            cv2.imshow('Video', cv2.resize(frame,None,fx=1,fy=1,interpolation = cv2.INTER_CUBIC))
                        
            cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                                            cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 230), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(100) & 0xFF == 27:
                        cv2.destroyAllWindows()
                        app.destroy()
                        sys.exit()
            
                
def Fisher(people_folder, shape):
        try:
          people = [person for person in os.listdir(people_folder)]
        except:
            messagebox.showinfo("Train Error","Please add at least one person to the system")
        print ("This are the people in the Recognition System:")
        for person in people:
            print ("-" + person)
        recognizer1 = cv2.face.createFisherFaceRecognizer()
        recognizer2 = cv2.face.createFisherFaceRecognizer()       
        threshold = 300
        images = []
        labels1 = []
        labels2 = []
        labels_people = {}
        labels_age={}
        path=people_folder
        for i, person in enumerate(people):
            p_age= os.path.split(path+person)[-1].split('.')[1]
            p=person
            person=os.path.split(path+person)[-1].split('.')[0]
            labels_people[i] = person
            labels_age[i] = p_age
            for image in os.listdir(people_folder + p):           
                images.append(cv2.imread(people_folder + p + '/' + image, 0))
                labels1.append(i) 
                labels2.append(i)  
                                    
        try:
            recognizer1.train(images, np.array(labels1))
            recognizer2.train(images, np.array(labels2))
            if not os.path.exists('recognizer1Fisher'):
                os.mkdir('recognizer1Fisher')
            if not os.path.exists('recognizer2Fisher'):  
                os.mkdir('recognizer2Fisher')
            recognizer1.save('recognizer1Fisher/trainingData.yml')
            recognizer2.save('recognizer2Fisher/trainingData.yml')
        except:
            messagebox.showinfo("OpenCV Error", "Do you have at least two people in the database?")
            app.destroy()
            sys.exit()
        video = VideoCamera()
        while True:
            frame = video.get_frame()
            faces_coord = detector.detect(frame, False)
            if len(faces_coord):
                frame, faces_img = get_images(frame, faces_coord, shape)
                for i, face_img in enumerate(faces_img):
                    if __version__ == "3.1.0":
                        collector = cv2.face.MinDistancePredictCollector()
                        recognizer1.predict(face_img, collector)
                        recognizer2.predict(face_img, collector)
                        conf = collector.getDist()
                        pred = collector.getLabel()
                    else:
                        pred, conf = recognizer1.predict(face_img)
                        pred, conf = recognizer2.predict(face_img)
                    print ("Prediction: " + str(pred))
                    print ('Confidence: ' + str(round(conf)))
                    print ('Threshold: ' + str(threshold))
                    
                    
                    if conf < threshold:
                        cv2.putText(frame, "Name: "+labels_people[pred].capitalize(),
                                    (faces_coord[i][0], faces_coord[i][1] - 20),
                                    cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                    cv2.LINE_AA)
                        cv2.putText(frame, "Age: "+labels_age[pred].capitalize(),
                                    (faces_coord[i][0], faces_coord[i][1] + 2),
                                    cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                    cv2.LINE_AA)
                        
                        for face in faces_coord:
                            (x,y,w,h)= face
                            newimg=frame[y:y+h,x:x+w]
                            newimg=cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
                            result=network.predict(newimg)
                            if result is not None:
                                result[0][2]-=0.15
                                result[0][4]-=0.15
                                if result[0][3] >0.06:
                                    result[0][3]+=0.4
                                maxindex=np.argmax(result[0]) 
                                cv2.putText(frame, EMOTIONS[maxindex],(faces_coord[i][0],
                                                  faces_coord[i][1] + 80),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                        cv2.LINE_AA)
                            cv2.imshow('Video', cv2.resize(frame,None,fx=1,fy=1,interpolation = cv2.INTER_CUBIC))
                                                    
                    else:
                        cv2.putText(frame, "Unknown",
                                    (faces_coord[i][0], faces_coord[i][1]),
                                    cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                    cv2.LINE_AA)
                        for face in faces_coord:
                            (x,y,w,h)= face
                            newimg=frame[y:y+h,x:x+w]
                            newimg=cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
                            result=network.predict(newimg)
                            if result is not None:
                                result[0][2]-=0.15
                                result[0][4]-=0.15
                                if result[0][3] >0.06:
                                    result[0][3]+=0.4
                                maxindex=np.argmax(result[0]) 
                                cv2.putText(frame, EMOTIONS[maxindex],(faces_coord[i][0],
                                                  faces_coord[i][1] + 80),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                        cv2.LINE_AA)
                            cv2.imshow('Video', cv2.resize(frame,None,fx=1,fy=1,interpolation = cv2.INTER_CUBIC))
    
            cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0,230), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(100) & 0xFF == 27:
                cv2.destroyAllWindows()
                app.destroy()
                sys.exit()
        
        
def Lbph(people_folder, shape):
        try:
          people = [person for person in os.listdir(people_folder)]
        except:
            messagebox.showinfo("Train Error","Please add at least one person to the system")
        print ("This are the people in the Recognition System:")
        for person in people:
            print ("-" + person)
        
        recognizer1 = cv2.face.createLBPHFaceRecognizer()
        recognizer2 = cv2.face.createLBPHFaceRecognizer()
        threshold = 105 
        images = []
        labels1 = []
        labels2 = []
        labels_people = {}
        labels_age={}
        path=people_folder
        for i, person in enumerate(people):
            p_age= os.path.split(path+person)[-1].split('.')[1]
            p=person
            person=os.path.split(path+person)[-1].split('.')[0]
            labels_people[i] = person
            labels_age[i] = p_age
            for image in os.listdir(people_folder + p):           
                images.append(cv2.imread(people_folder + p + '/' + image, 0))
                labels1.append(i) 
                labels2.append(i)  
        try:
            recognizer1.train(images, np.array(labels1))
            recognizer2.train(images, np.array(labels2))
            if not os.path.exists('recognizer1LBPH'): 
              os.mkdir('recognizer1LBPH')
            if not os.path.exists('recognizer2LBPH'): 
              os.mkdir('recognizer2LBPH')
            recognizer1.save('recognizer1LBPH/trainingData.yml')
            recognizer2.save('recognizer2LBPH/trainingData.yml')
        except:
            messagebox.showinfo("OpenCV Error", "Do you have at least one people in the database?")
            app.destroy()
            sys.exit()
        video = VideoCamera()
        while True:
            frame = video.get_frame()
            faces_coord = detector.detect(frame, False)
            if len(faces_coord):
                frame, faces_img = get_images(frame, faces_coord, shape)
                for i, face_img in enumerate(faces_img):
                    if __version__ == "3.1.0":
                        collector = cv2.face.MinDistancePredictCollector()
                        recognizer1.predict(face_img, collector)
                        recognizer2.predict(face_img, collector)
                        conf = collector.getDist()
                        pred = collector.getLabel()
                    else:
                        pred, conf = recognizer1.predict(face_img)
                        pred, conf = recognizer2.predict(face_img)
                    print ("Prediction: " + str(pred))
                    print ('Confidence: ' + str(round(conf)))
                    print ('Threshold: ' + str(threshold))
                    if conf < threshold:
                        cv2.putText(frame, "Name: "+labels_people[pred].capitalize(),
                                    (faces_coord[i][0], faces_coord[i][1] - 20),
                                    cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                    cv2.LINE_AA)
                        cv2.putText(frame, "Age: "+labels_age[pred].capitalize(),
                                    (faces_coord[i][0], faces_coord[i][1] + 2),
                                    cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                    cv2.LINE_AA)
                        for face in faces_coord:
                            (x,y,w,h)= face
                            newimg=frame[y:y+h,x:x+w]
                            newimg=cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
                            result=network.predict(newimg)
                            if result is not None:
                                result[0][2]-=0.15
                                result[0][4]-=0.15
                                if result[0][3] >0.06:
                                    result[0][3]+=0.4
                                maxindex=np.argmax(result[0]) 
                                cv2.putText(frame, EMOTIONS[maxindex],(faces_coord[i][0],
                                                  faces_coord[i][1] + 200),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                        cv2.LINE_AA)
                            cv2.imshow('Video', cv2.resize(frame,None,fx=1,fy=1,interpolation = cv2.INTER_CUBIC))
                    else:
                        cv2.putText(frame, "Unknown",
                                    (faces_coord[i][0], faces_coord[i][1]),
                                    cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                    cv2.LINE_AA)
                        for face in faces_coord:
                            (x,y,w,h)= face
                            newimg=frame[y:y+h,x:x+w]
                            newimg=cv2.resize(newimg, (48,48), interpolation = cv2.INTER_CUBIC) / 255.
                            result=network.predict(newimg)
                            if result is not None:
                                result[0][2]-=0.15
                                result[0][4]-=0.15
                                if result[0][3] >0.06:
                                    result[0][3]+=0.4
                                maxindex=np.argmax(result[0]) 
                                cv2.putText(frame, EMOTIONS[maxindex],(faces_coord[i][0],
                                                  faces_coord[i][1] + 200),
                                cv2.FONT_HERSHEY_PLAIN, 1.7, (230, 0, 0), 2,
                                        cv2.LINE_AA)
                            cv2.imshow('Video', cv2.resize(frame,None,fx=1,fy=1,interpolation = cv2.INTER_CUBIC))
    
            cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 5),
                        cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 0,230), 2)
            cv2.imshow('Video', frame)
            if cv2.waitKey(100) & 0xFF == 27:
                cv2.destroyAllWindows()
                app.destroy()
                sys.exit()
             
def Exit():
    app.destroy()
    sys.exit()
    
PEOPLE_FOLDER = "Face_Database/"
SHAPE = "ellipse"
LARGE_FONT1=("Prosto one", 30)
LARGE_FONT2=("Prosto one", 12)
class  Faceapp(tk.Tk): 
    def __init__(self, *args,  **kwargs):
        tk.Tk.__init__(self, *args,  **kwargs)
        self.geometry('900x500')
        
        container=tk.Frame(self)
        container.pack(side="top", fill="both" ,expand= True)
        
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)
        
        self.frames={ }
        
        for i in (Startpage,Pagetwo):
        
           frame=i(container,self)        
           self.frames[i]=frame
           frame.grid(row=0 , column=0, sticky="nsew")
        self.show_frame(Startpage)
        
    def show_frame(self,cont):
        frame=self.frames[cont]
        frame.tkraise()
        
class Startpage(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        self.configure(bg='#82E3BE')       
        label1 =tk.Label(self, text="FACE RECOGNITION SYSTEM",width=40,font=LARGE_FONT1)
        label1.configure(bg='#82E3BE')
        label1.pack(pady=20,padx=20)
        
        label30 =tk.Label(self, text="Add People",width=15,font=LARGE_FONT2)
        label30.configure(bg='#82E3BE')
        label30.place(x=80,y=300)   
                
        label31 =tk.Label(self, text="Start Recognizer",width=15,font=LARGE_FONT2)
        label31.configure(bg='#82E3BE')
        label31.place(x=385,y=300)
        
        label32 =tk.Label(self, text="Exit",width=15,font=LARGE_FONT2)
        label32.configure(bg='#82E3BE')
        label32.place(x=680,y=300)
        
        self.xone=Image.open("images/add.jpg")
        self.xtwo=Image.open("images/start.jpg")
        self.xthree=Image.open("images/exit.jpg")
        self.one=ImageTk.PhotoImage(self.xone)
        self.two=ImageTk.PhotoImage(self.xtwo)
        self.three=ImageTk.PhotoImage(self.xthree)
        
        button1=tk.Button(self,image=self.one,command=lambda:
            add_person(PEOPLE_FOLDER, SHAPE))
        button1.place(x=100,y=160) 
          
        button3=tk.Button(self,image=self.two,command=lambda: 
            controller.show_frame(Pagetwo))
        button3.place(x=400,y=160)    
        
        button81=tk.Button(self,image=self.three,command=lambda: 
            Exit())
        button81.place(x=700,y=160)
  
class Pagetwo(tk.Frame):
    def __init__(self,parent,controller):
        tk.Frame.__init__(self,parent)
        self.configure(bg='#82E3BE')               
        label3 =tk.Label(self, text="FACE RECOGNITION SYSTEM",width=40,font=LARGE_FONT1)
        label3.pack(pady=10,padx=10)
        label3.configure(bg='#82E3BE')
                         
        self.xfour=Image.open("images/eigen.jpg")
        self.xfive=Image.open("images/fisher.jpg")
        self.xsix=Image.open("images/lbph.jpg")
        self.xseven=Image.open("images/home.jpg")
        self.four=ImageTk.PhotoImage(self.xfour)
        self.five=ImageTk.PhotoImage(self.xfive)
        self.six=ImageTk.PhotoImage(self.xsix)
        self.seven=ImageTk.PhotoImage(self.xseven)
        
        label23 =tk.Label(self, text="Choose Algorithm",font=("Prosto one",23))
        label23.place(x=215,y=96)
        label23.configure(bg='#82E3BE')
        label33 =tk.Label(self, text="Home",width=15,font=LARGE_FONT2)
        label33.configure(bg='#82E3BE')
        label33.place(x=600,y=255)                  
        
        button4=tk.Button(self,image=self.four,command=lambda: 
            Eigen(PEOPLE_FOLDER, SHAPE))
        button4.place(x=270,y=150)
        
        button5=tk.Button(self,image=self.five,command=lambda: 
            Fisher(PEOPLE_FOLDER, SHAPE))
        button5.place(x=270,y=265) 
        
        button6=tk.Button(self,image=self.six,command=lambda: 
            Lbph(PEOPLE_FOLDER, SHAPE))
        button6.place(x=270,y=380) 
        
        button7=tk.Button(self,image=self.seven,command=lambda: 
            controller.show_frame(Startpage))
        button7.place(x=640,y=150)   
app=Faceapp()    
if __name__ == '__main__':   
   app.mainloop()     


         
        
   