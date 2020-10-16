import cv2
import os
import numpy as np
from tkinter import * 
from tkinter.filedialog import *
from tkinter.messagebox import *
import tkinter as tk
from tkinter import ttk
from matplotlib import pyplot as plt
def save_path():
    path = os.getcwd() 
    os.chdir(path)
class Pointprocessing():
    def imageNegative():
        file = askopenfilename(filetypes =[('Image File','.jpg')])
        img = cv2.imread(file, 1)
        
        img_not = cv2.bitwise_not(img)
        res = np.hstack((img, img_not)) 
        cv2.imshow("Image negetive",res)

        save_path()
        filename = 'Image Negative.jpg'
        cv2.imwrite(filename, res)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # imageNegative()
    def contrastStreching():
        file = askopenfilename(filetypes =[('Image File','.jpg')])

        img = cv2.imread(file)
        xp = [0, 64, 128, 192, 255]
        fp = [0, 16, 128, 240, 255]
        x = np.arange(256)
        table = np.interp(x, xp, fp).astype('uint8')
        img_not = cv2.LUT(img, table)
        res = np.hstack((img, img_not)) 
        cv2.imshow("Contrast streching", res)
        save_path()   
        
        filename = 'Contrast Streching.jpg'
        cv2.imwrite(filename, res)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    # contrastStreching()
    def histrogramEqualization():
        file = askopenfilename(filetypes =[('Image File','.jpg')])
        img = cv2.imread(file,0) 
        # gray = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        equ = cv2.equalizeHist(img) 
        res = np.hstack((img , equ)) 
        cv2.imshow('Histrogram Equalization', res)
        save_path() 
        filename = 'Histrogram Equalization.jpg'
        cv2.imwrite(filename, res) 
        cv2.waitKey(0) 
        cv2.destroyAllWindows() 
    # histrogramEqualization()
    def powerLow():
        file = askopenfilename(filetypes =[('Image File','.jpg')])
        img = cv2.imread(file)
        # Apply Gamma=2.2 on the normalised image and then multiply by scaling constant (For 8 bit, c=255)
        gamma_two_point_two = np.array(255*(img/255)**2.2,dtype='uint8')
        # Similarly, Apply Gamma=0.4 
        gamma_point_four = np.array(255*(img/255)**0.4,dtype='uint8')
        img3 = cv2.hconcat([gamma_two_point_two])
        res3 = np.hstack((img,img3))
        cv2.imshow('Power Low Gamma2.2',res3)
        save_path()
        filename = 'Power Low Gamma2.2.jpg'
        cv2.imwrite(filename, res3) 
        
        img4 = cv2.hconcat([gamma_point_four])
        res4 = np.hstack((img, img4)) 
        cv2.imshow('Power Low Gamma 0.4',res4)
        save_path()
        filename = 'Power Low Gamma 0.4.jpg'
        cv2.imwrite(filename, res4)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    # powerLow()
    def intensitylevelslicing():
        file = askopenfilename(filetypes =[('Image File','.jpg')])
        img = cv2.imread(file)
        imag = cv2.resize()
        row, column = imag.shape
        img1 = np.zeros((row,column),dtype = 'uint8')
        min_range = 10
        max_range = 60
        for i in range(row):
            for j in range(column):
                if img[i,j]>min_range and img[i,j]<max_range:
                    img1[i,j] = 255
                else:
                    img1[i,j] = 0
        res = np.hstack((img,img1))
        cv2.imshow('sliced image', res)
        save_path()
        filename = 'Intensity level slicing.jpg'
        cv2.imwrite(filename, res) 
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    # intensitylevelslicing()
class Neighborhoodprocessing(): 
    def imagesmoothing():
        file = askopenfilename(filetypes =[('Image File','.jpg')])
        img = cv2.imread(file)
        blur = cv2.blur(img,(5,5))
        res = np.hstack((img, blur)) 
        cv2.imshow("Image Smoothing",res)
        save_path()
        filename = 'Image Smoothing.jpg'
        cv2.imwrite(filename, res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # imagesmoothing()
    def gaussian():
        file = askopenfilename(filetypes =[('Image File','.jpg')])
        img = cv2.imread(file)
        gaus = cv2.GaussianBlur(img,(5,5),0)
        res = np.hstack((img, gaus)) 

        cv2.imshow("Gaussian ",res)
        save_path()
        filename = 'Image Smoothing.jpg'
        cv2.imwrite(filename, res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def median():
        file = askopenfilename(filetypes =[('Image File','.jpg')])
        img = cv2.imread(file)
        med = cv2.medianBlur(img,5)
        res = np.hstack((img, med)) 
        cv2.imshow("Median",res)
        save_path()
        filename = 'Median.jpg'
        cv2.imwrite(filename, res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # median()
    def laplacian():
        file = askopenfilename(filetypes =[('Image File','.jpg')])
        img0 = cv2.imread(file)
        gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(img0, cv2.CV_64F)
        res = np.hstack((laplacian, img0)) 
        save_path()
        filename = 'Sharpanning image.jpg'
        i = cv2.imwrite(filename, res)
        # cv2.imshow("Sharpanning image",)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # laplacian()
    def highboost():
        file = askopenfilename(filetypes =[('Image File','.jpg')])
        image = cv2.imread(file, cv2.COLOR_BGR2GRAY)
        gauss_mask = cv2.GaussianBlur(image, (9, 9), 10.0)
        image_sharp = cv2.addWeighted(image, 2, gauss_mask, -1, 0)
        #High pass Kernel 3x3
        kernel = np.array([[-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1]])
        image_hpf = cv2.filter2D(image, -1, kernel)
        res = np.hstack((image,image_hpf)) 
        cv2.imshow("Highboost Filter", res)
        save_path()
        filename = 'highboost image.jpg'
        cv2.imwrite(filename, res)
        cv2.waitKey(0)
#<-=-==-=-=-=-=-=-=-==-=-=-=-======-=-=-=-=-=-=-=-==-=-=-=-=-=-=-=>
main = Tk()
main.title("Image transformation")
main.iconbitmap()
main  
l = Label(main, text = "Choose your option for Point Processing ") 
l.config(font =("Courier", 14))
 
b1 = Button(main, text = "Image Negative", command=lambda: Pointprocessing.imageNegative()) 
b2 = Button(main, text = "Contrast Streching", command=lambda: Pointprocessing.contrastStreching()) 
b3 = Button(main, text = "HistrogramEqualization", command=lambda: Pointprocessing.histrogramEqualization()) 
b4 = Button(main, text = "Power Low", command=lambda: Pointprocessing.powerLow()) 
b5 = Button(main, text = "Intensity level slicing", command=lambda: Pointprocessing.intensitylevelslicing()) 
l.pack() 
b1.pack(side=TOP)
b2.pack(side=TOP)
b3.pack(side=TOP)
b4.pack(side=TOP)
b5.pack(side=TOP)

l = Label(main, text = "Choose your option for Neighborhood processing ") 
l.config(font =("Courier", 14))
b1 = Button(main, text = "Image Smoothing", command=lambda: Neighborhoodprocessing.imagesmoothing()) 
b2 = Button(main, text = "Gaussian image", command=lambda: Neighborhoodprocessing.gaussian()) 
b3 = Button(main, text = "Median image ", command=lambda: Neighborhoodprocessing.median()) 
b4 = Button(main, text = "Sharapnning Image", command=lambda: Neighborhoodprocessing.laplacian()) 
b5 = Button(main, text = "highboost  lmage", command=lambda: Neighborhoodprocessing.highboost()) 
l.pack() 
b1.pack(side=TOP)
b2.pack(side=TOP)
b3.pack(side=TOP)
b4.pack(side=TOP)
b5.pack(side=TOP)
b2 = Button(main, text = "Exit",command = main.destroy)  
b2.pack()
# main.config(menu=menubar)
main.mainloop()
