import cv2
import os
import numpy as np
from tkinter import * 
import tkinter as tk
from tkinter.filedialog import *
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
import time
tim = time.time_ns()

try:
    outpur_dir = os.mkdir('OUTPUT')
except:
    print("Directory alreafy exist")

def readFile(isGrey = 1):
    global file
    file = askopenfilename(filetypes =[('Image File',['.jpeg', '.jpg', '.png'])])
    try :
        img = cv2.imread(file, isGrey)
        return img
    except :
        print('Plese Select Image')

def saveImage(img, modifiedImg, opName="imgtransform"):
    try:
        res = np.hstack((img, modifiedImg)) 
        
        targetFileName = (Path(file)).stem + "_" + str(tim) + "_" + opName  + (Path(file)).suffix  
        print(targetFileName)
        cv2.imwrite(targetFileName, res)
        Image.open(targetFileName).show()

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)

class Pointprocessing():
    def imageNegative():

        img = readFile()
        modifiedImg = cv2.bitwise_not(img)
        saveImage(img, modifiedImg, "img_negative")     

    def contrastStreching():
        img = readFile()      
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        modifiedImg = cv2.LUT(img, table)
        saveImage(img , modifiedImg,'contrastStreching')

    def histrogramEqualization():
        img = readFile(0) 
        modifiedImg = cv2.equalizeHist(img)
        saveImage(img, modifiedImg,'histrogramEqualization') 

    def powerLow():
        img = readFile()
        powerlow_level= float(x_input())
        gamma_two_point_two = np.array(255*((img/255)**powerlow_level),dtype='uint8')
        modifiedImg1 = cv2.hconcat([gamma_two_point_two])
        saveImage(img ,modifiedImg1,'powerLow')

    def intensitylevelslicing():
        img = readFile(0)
        row, column = img.shape
        modifiedImg = np.zeros((row,column),dtype = 'uint8')
        min_range = int(x_input())
        max_range = int(y_input())
        for i in range(row):
            for j in range(column):
                if img[i,j]>min_range and img[i,j]<max_range:
                    modifiedImg[i,j] = 255
                else:
                    modifiedImg[i,j] = 0
        saveImage(img, modifiedImg,'intensitylevelslicing')

class Neighborhoodprocessing(): 
    def imageSmoothing():
        img = readFile()
        modifiedImg = cv2.blur(img,(int(x_input()),int(y_input())))
        saveImage(img, modifiedImg,'imageSmoothing')

    def gaussian():
        img = readFile()
        modifiedImg = cv2.GaussianBlur(img,(int(x_input()),int(y_input())),0)
        saveImage(img, modifiedImg,'gaussian')

    def median():
        img = readFile()
        median_number=int(x_input())
        modifiedImg = cv2.medianBlur(img,median_number)
        saveImage(img, modifiedImg,'median')

    def laplacian():
        img = readFile()
        modifiedImg = cv2.Laplacian(img, cv2.CV_64F)
        saveImage(img, modifiedImg,'laplacian') 

    def highboost():
        img = readFile()
        highboost= int(x_input())
        kernel = np.array([[-1, -1, -1],
                        [-1,  highboost, -1],
                        [-1, -1, -1]])
        modifiedImg = cv2.filter2D(img, -1, kernel)
        saveImage(img , modifiedImg,'highboost')
        
#<-=-==-=-=-=-=-=-=-==-=-=-=-======-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=>
main = Tk()
main.title("Image transformation")
main.iconbitmap()

l = Label(main, text = """('--'  keep empty the box, 'Level'- Fill one box only,\n'Cordinate'- Fill cordinates in box,'Min,Max'- Fill Minimum and Maximum value)""") 
l.config(font =("Courier", 10))
l.pack()

var1 = IntVar()
var1.set("Level Or X-cordiante Or Min")
label1 = Label(main,textvariable=var1,height = 2)
label1.pack()

ID=IntVar()
box1=Entry(main,bd=4,textvariable=ID)
box1.pack()

var2 = IntVar()
var2.set("Y-cordinate Or Max")
label2 = Label(main,textvariable=var2,height = 2)
label2.pack()

ID=IntVar()
box2=Entry(main,bd=4,textvariable=ID)
box2.pack()

def y_input():
    a = box2.get()
    return a
def x_input():
    a = box1.get()
    return a

l1 = Label(main, text = "Choose your option for Point Processing ") 
l1.config(font =("Courier", 14))
 
b1 = Button(main, text = "Image Negative -- ", command=lambda: Pointprocessing.imageNegative()) 
b2 = Button(main, text = "Contrast Streching -- ", command=lambda: Pointprocessing.contrastStreching()) 
b3 = Button(main, text = "HistrogramEqualization -- ", command=lambda: Pointprocessing.histrogramEqualization()) 
b4 = Button(main, text = "Power Low , (Level)", command=lambda : Pointprocessing.powerLow()) 
b5 = Button(main, text = "Intensity level slicing (Min, Max)", command=lambda: Pointprocessing.intensitylevelslicing()) 

l1.pack() 
b1.pack(side=TOP)
b2.pack(side=TOP)
b3.pack(side=TOP)
b4.pack(side=TOP)
b5.pack(side=TOP)

l2 = Label(main, text = "Choose your option for Neighborhood processing ") 
l2.config(font =("Courier", 14))
b1 = Button(main, text = "Image Smoothing (Cordinates)", command=lambda: Neighborhoodprocessing.imageSmoothing()) 
b2 = Button(main, text = "Gaussian Image (Cordinates)", command=lambda: Neighborhoodprocessing.gaussian()) 
b3 = Button(main, text = "Median Image (Level) ", command= lambda:  Neighborhoodprocessing.median()) 
b4 = Button(main, text = "Laplacian Sharapnning Image --", command=lambda: Neighborhoodprocessing.laplacian()) 
b5 = Button(main, text = "Highboost  Image (Level)", command=lambda: Neighborhoodprocessing.highboost()) 

l2.pack() 
b1.pack(side=TOP)
b2.pack(side=TOP)
b3.pack(side=TOP)
b4.pack(side=TOP)
b5.pack(side=TOP)

b2 = Button(main, text = "Exit",command = main.destroy)  
b2.pack()
main.mainloop()
