import numpy as np
import cv2

from tkinter import filedialog
from tkinter import Tk, Button, Label, Frame, Menu, TOP, HORIZONTAL, LEFT, X, messagebox, Scale
from PIL import Image
from PIL import ImageTk

import fft
import image_utils
import filtering

class ipGUI:
    def __init__(self, master):
        self.master = master
        self.master.minsize(width=800, height=600)

        menu = Menu(self.master)
        master.config(menu=menu)

        # ***** Main Menu *****
        # *** file menu ***
        fileMenu = Menu(menu)
        menu.add_cascade(label='File', menu=fileMenu)
        fileMenu.add_command(label="Open", command=self.openImage)
        fileMenu.add_command(label="Load Blur Kernel", command=self.loadKernel)
        fileMenu.add_command(label="Save", command=self.saveImage)
        fileMenu.add_command(label="Exit", command=master.destroy)

        # *** edit menu ***
        editMenu = Menu(menu)
        menu.add_cascade(label='Space', menu=editMenu)
        editMenu.add_command(label="Histogram Equalization", command=self.histWrap)
        editMenu.add_command(label="Gamma Correction", command=self.gammaCorWrap)
        editMenu.add_command(label="Log Transform", command=self.logTranWrap)
        editMenu.add_command(label="Sharpen", command=self.sharpWrap)
        editMenu.add_command(label="Cartoonify", command=self.cartoonifyWrap)

        # *** blur menu ***
        blurMenu = Menu(editMenu)
        editMenu.add_cascade(label='Blur', menu=blurMenu)
        blurMenu.add_command(label="Box", command=self.boxBlurWrap)
        blurMenu.add_command(label="Gaussian", command=self.gaussianBlurWrap)
        blurMenu.add_command(label="Median", command=self.medianBlurWrap)

        # *** frequency filtering ***
        freqMenu = Menu(menu)
        menu.add_cascade(label='Frequency', menu=freqMenu)
        freqMenu.add_command(label="DFT", command=self.dftWrap)
        freqMenu.add_command(label="Load Mask", command=self.freqMaskWrap)  

        # *** Mask Menu ***
        maskMenu = Menu(freqMenu)
        freqMenu.add_cascade(label='Create Mask', menu=maskMenu)
        maskMenu.add_command(label='Low Pass', command=self.lpmWrap)
        maskMenu.add_command(label='High Pass', command=self.hpmWrap)
        maskMenu.add_command(label='Band Pass', command=self.bppmWrap)
        maskMenu.add_command(label='Band Stop', command=self.bspmWrap)

        # *** frequency filtering ***
        restorationMenu = Menu(menu)
        menu.add_cascade(label='Restoration', menu=restorationMenu)
        restorationMenu.add_command(label="Full Inverse", command=self.fullInverseWrap)
        restorationMenu.add_command(label="Radially Limited Inverse", command=self.truncatedInverseWrap)
        restorationMenu.add_command(label="Approximate Weiner", command=self.approximateWeinerWrap)
        restorationMenu.add_command(label="Constrained LS", command=self.contrainedLSWrap)

        # ***** Toolbar *****
        toolbar = Frame(master, bg="grey")
        undoButton = Button(toolbar, text="Undo", command=self.undoFunc)
        undoButton.pack(side=LEFT)
        origButton = Button(toolbar, text="Original", command=self.origFunc)
        origButton.pack(side=LEFT)
        toolbar.pack(side=TOP, fill=X)

        # ***** Image Display Area *****
        self.frame = Frame(self.master)
        self.frame.pack()
        self.panel = Label(self.frame)
        self.panel.pack(padx=10, pady=10)
        self.img = None
        self.origImg = None
        self.prevImg = None

        # ***** Gamma Controls *****
        self.gammaFrame = Frame(self.master)
        self.gammaSlider = Scale(self.gammaFrame, from_=0.1, to=2, orient=HORIZONTAL, resolution=0.1)
        self.gammaSlider.pack(side=TOP)
        self.gammaExitButton = Button(self.gammaFrame, text="Exit", command=self.gammaFrame.pack_forget)
        self.gammaExitButton.pack(side=TOP)

        # ***** Box Blur Controls *****
        self.boxFrame = Frame(self.master)
        self.boxSlider = Scale(self.boxFrame, from_=1, to=5, orient=HORIZONTAL)
        self.boxSlider.pack(side=TOP)
        self.boxExitButton = Button(self.boxFrame, text="Exit", command=self.boxFrame.pack_forget)
        self.boxExitButton.pack(side=TOP)

        # ***** Truncated Inverse Controls *****
        self.truncatedInverseFrame = Frame(self.master)
        self.truncatedInverseSlider = Scale(self.truncatedInverseFrame, from_=-3, to=2, orient=HORIZONTAL, resolution=0.1)
        self.truncatedInverseSlider.pack(side=TOP)
        self.truncatedInverseExitButton = Button(self.truncatedInverseFrame, text="Exit", command=self.truncatedInverseFrame.pack_forget)
        self.truncatedInverseExitButton.pack(side=TOP)

        # ***** Weiner Controls *****
        self.weinerFrame = Frame(self.master)
        self.weinerSlider = Scale(self.weinerFrame, from_=-3, to=2, orient=HORIZONTAL, resolution=0.1)
        self.weinerSlider.pack(side=TOP)
        self.weinerExitButton = Button(self.weinerFrame, text="Exit", command=self.weinerFrame.pack_forget)
        self.weinerExitButton.pack(side=TOP)

        # ***** CLS Controls *****
        self.clsFrame = Frame(self.master)
        self.clsSlider = Scale(self.clsFrame, from_=-3, to=2, orient=HORIZONTAL, resolution=0.1)
        self.clsSlider.pack(side=TOP)
        self.clsExitButton = Button(self.clsFrame, text="Exit", command=self.clsFrame.pack_forget)
        self.clsExitButton.pack(side=TOP)

        # ***** DFT Display Area ******
        self.dftFrame = Frame(self.master)
        self.magPanel = Label(self.dftFrame)
        self.magPanel.pack(padx=10, pady=10,side=TOP)
        self.freqPanel = Label(self.dftFrame)
        self.freqPanel.pack(padx=10, pady=10,side=TOP)
        self.dftExitButton = Button(self.dftFrame, text="Exit", command=lambda: self.displayImg(np.array(self.img)))
        self.dftExitButton.pack(side=TOP, fill=X)

        # ***** Low Pass Mask Creation *****
        self.lpmFrame = Frame(self.master)
        self.lpmPanel = Label(self.lpmFrame)
        self.lpmPanel.pack(padx=10, pady=10,side=TOP)
        self.lpmSubButton = Button(self.lpmFrame, text="submit")
        self.lpmSubButton.pack(side=TOP)
        self.lpmExitButton = Button(self.lpmFrame, text="Exit", command=lambda: self.displayImg(np.array(self.img)))
        self.lpmExitButton.pack(side=TOP)
        self.lpmSlider = Scale(self.lpmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)

        # ***** High Pass Mask Creation *****
        self.hpmFrame = Frame(self.master)
        self.hpmPanel = Label(self.hpmFrame)
        self.hpmPanel.pack(padx=10, pady=10,side=TOP)
        self.hpmSubButton = Button(self.hpmFrame, text="submit")
        self.hpmSubButton.pack(side=TOP)
        self.hpmExitButton = Button(self.hpmFrame, text="Exit", command=lambda: self.displayImg(np.array(self.img)))
        self.hpmExitButton.pack(side=TOP)
        self.hpmSlider = Scale(self.hpmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)

        # ***** Band Pass Mask Creation *****
        self.bppmFrame = Frame(self.master)
        self.bppmPanel = Label(self.bppmFrame)
        self.bppmPanel.pack(padx=10, pady=10,side=TOP)
        self.bppmSubButton = Button(self.bppmFrame, text="submit")
        self.bppmSubButton.pack(side=TOP)
        self.bppmExitButton = Button(self.bppmFrame, text="Exit", command=lambda: self.displayImg(np.array(self.img)))
        self.bppmExitButton.pack(side=TOP)
        self.bppmSliderLow = Scale(self.bppmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)
        self.bppmSliderHigh = Scale(self.bppmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)

        # ***** Band Pass Mask Creation *****
        self.bspmFrame = Frame(self.master)
        self.bspmPanel = Label(self.bspmFrame)
        self.bspmPanel.pack(padx=10, pady=10,side=TOP)
        self.bspmSubButton = Button(self.bspmFrame, text="submit")
        self.bspmSubButton.pack(side=TOP)
        self.bspmExitButton = Button(self.bspmFrame, text="exit", command=lambda: self.displayImg(np.array(self.img)))
        self.bspmExitButton.pack(side=TOP)
        self.bspmSliderLow = Scale(self.bspmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)
        self.bspmSliderHigh = Scale(self.bspmFrame, from_=1, to=2, orient=HORIZONTAL, resolution=1)


    def displayImg(self, img):
        # input image in RGB
        self.frame.pack()
        self.dftFrame.pack_forget()
        self.lpmFrame.pack_forget()
        self.hpmFrame.pack_forget()
        self.bppmFrame.pack_forget()
        self.bspmFrame.pack_forget()
        self.img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(self.img)

        self.panel.configure(image=imgtk)
        self.panel.image = imgtk

    def openImage(self):
        # can change the image
        path = filedialog.askopenfilename()
        if len(path) > 0:
            imgRead = cv2.imread(path)
            if imgRead is not None:
                imgRead = cv2.cvtColor(imgRead, cv2.COLOR_BGR2RGB)
                self.origImg = Image.fromarray(imgRead)
                self.prevImg = Image.fromarray(imgRead)
                self.displayImg(imgRead)
            else:
                raise ValueError("Not a valid image")
        else:
            raise ValueError("Not a valid path")

    def loadKernel(self):
        path = filedialog.askopenfilename()
        if len(path) > 0:
            self.filter_kernel = np.mean(cv2.imread(path), axis=-1)
            self.filter_kernel = self.filter_kernel/np.sum(self.filter_kernel)
        else:
            raise ValueError("Not a valid path")

    def saveImage(self):
        if self.img is not None:
            toSave = filedialog.asksaveasfilename()
            self.img.save(toSave)
        else:
            messagebox.showerror(title="Save Error", message="No image to be saved!")

    def histWrap(self):
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = image_utils.histEqlFunc(img)
        self.displayImg(imgNew)

    # Gamma Correction
    def gammaCallback(self, event, img):
        gamma = self.gammaSlider.get()
        self.prevImg = self.img
        imgNew = image_utils.gammaCorFunc(img, 255.0/(255.0**gamma), gamma)
        self.displayImg(imgNew)

    def truncatedInverseCallback(self, event, img):
        th = 10**self.truncatedInverseSlider.get()
        imgNew = self.truncatedInverseFilter(th)
        self.displayImg(imgNew)

    def weinerCallback(self, event, img):
        gamma = 10**self.weinerSlider.get()
        imgNew = self.weinerFilter(gamma)
        self.displayImg(imgNew)

    def clsCallback(self, event, img):
        K = 10**self.clsSlider.get()
        imgNew = self.clsFilter(K)
        self.displayImg(imgNew)

    def gammaCorWrap(self):
        img = np.array(self.img)
        self.gammaFrame.pack()
        self.gammaSlider.set(1.0)
        self.gammaSlider.bind("<ButtonRelease-1>", lambda event, img=img: self.gammaCallback(event, img))

    def logTranWrap(self):
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = image_utils.logTranFunc(img, 255.0/np.log10(256))
        self.displayImg(imgNew)

    def sharpWrap(self):
        # Sharpen Wrapper
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = image_utils.sharpFunc(img)
        self.displayImg(imgNew)

    def boxBlurWrap(self):
        # Box Blur Wrapper
        img = np.array(self.img)
        self.prevImg = self.img
        self.boxFrame.pack()
        self.boxSlider.set(1)
        self.boxSlider.bind("<ButtonRelease-1>", lambda event, img=img: self.displayImg
                            (image_utils.boxBlurFunc(event, img, 2*self.boxSlider.get()+1)))

    def gaussianBlurWrap(self):
        # Gaussian Blur Wrapper
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = image_utils.gaussianBlurFunc(img)
        self.displayImg(imgNew)

    def medianBlurWrap(self):
        # Median Blur Wrapper
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = image_utils.medianBlurFunc(img)
        self.displayImg(imgNew)

    def cartoonifyWrap(self):
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = image_utils.cartoonifyFunc(img)
        self.displayImg(imgNew)

    def undoFunc(self):
        self.img = self.prevImg
        self.displayImg(np.array(self.img))

    def origFunc(self):
        self.prevImg = self.img
        self.img = self.origImg
        self.displayImg(np.array(self.img))

    def displayDFT(self, x_fft):
        m, n = x_fft.shape[:]
        self.frame.pack_forget()
        self.dftFrame.pack()

        x_mag = np.log10(np.absolute(x_fft))*255/np.log10(m*n*255)
        x_mag = Image.fromarray(x_mag)
        x_mag = x_mag.resize((min(256,m), min(int(256*n/m),n)), Image.ANTIALIAS)
        x_mag = ImageTk.PhotoImage(x_mag)

        x_freq = (np.angle(x_fft)%360)*255/360
        x_freq = Image.fromarray(x_freq)
        x_freq = x_freq.resize((min(256,m), min(int(256*n/m),n)), Image.ANTIALIAS)
        x_freq = ImageTk.PhotoImage(x_freq)

        self.magPanel.configure(image=x_mag)
        self.magPanel.image = x_mag

        self.freqPanel.configure(image=x_freq)
        self.freqPanel.image = x_freq

    def dftWrap(self):
        img = np.array(self.img)
        x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x_outp_img = fft.fft2d_img(x)
        self.displayDFT(x_outp_img)

    def freqMaskWrap(self):
        m,n = np.array(self.img).shape[:2]
        lm, ln = np.log2(m), np.log2(m)
        dm, dn = int(2**np.ceil(lm))+1, int(2**np.ceil(ln))+1
        messagebox.showinfo("Mask Size", "Mask Size should be ("+str(dm)+","+str(dn)+")")
        path = filedialog.askopenfilename()
        if len(path) > 0:
            maskRead = cv2.imread(path, 0)
            if(maskRead.shape != (dm,dn)):
                messagebox.showerror(title="Shape Error", message="Shape of mask and image don't match")
            else:
                self.prevImg = self.img
                self.display(image_utils.freqMask(np.array(self.img), maskRead))
        else:
            raise ValueError("Not a valid path")

    def maskWrap(self):
        img = np.array(self.img)
        x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        x_outp_img = fft.fft2d_img(x)
        x_mag = image_utils.xMagCreate(x_outp_img)

        return x_outp_img, x_mag

    def maskFinal(self, x_outp_img):
        self.prevImg = self.img
        img = np.array(self.img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        m = len(img)
        res = fft.ifft2d_img(x_outp_img)
        res = np.array(res*255/np.max(res),dtype=np.uint8)
        res = np.array([[[img[i][j][0], img[i][j][1], res[i][j]] for j in range(m)] for i in range(m)])
        res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
        self.displayImg(res)

    def lpmCallBack(self, event, slider, x_outp_img):
        m = len(x_outp_img)
        m2 = int((m-1)/2)
        r = slider.get()
        x_outp_copy = np.copy(x_outp_img)

        nCircle = np.array([[(i-m2)**2+(j-m2)**2 for j in range(m)] for i in range(m)])
        nCircle = np.where(nCircle>r**2)
        x_outp_copy[nCircle] = 1

        x_mag = image_utils.xMagCreate(x_outp_copy)

        self.lpmPanel.configure(image=x_mag)
        self.lpmPanel.image = x_mag

    def lpmFinal(self, x_outp_img):
        mx = len(x_outp_img)
        m2 = int((mx-1)/2)
        r = self.lpmSlider.get()
        nCircle = np.array([[(i-m2)**2+(j-m2)**2 for j in range(mx)] for i in range(mx)])
        nCircle = np.where(nCircle>r**2)
        x_outp_img[nCircle] = 0

        self.maskFinal(x_outp_img)


    def lpmWrap(self):
        x_outp_img, x_mag = self.maskWrap()
        m = len(x_outp_img)

        self.lpmPanel.configure(image=x_mag)
        self.lpmPanel.image = x_mag

        self.lpmSlider.configure(to=int(np.ceil(m/2)))
        self.lpmSlider.set(1)
        self.lpmSlider.pack(side=TOP)
        self.lpmSlider.bind("<ButtonRelease-1>", lambda event, slider=self.lpmSlider, x_outp_img=x_outp_img: self.lpmCallBack(event, slider, x_outp_img))

        self.lpmSubButton.configure(command=lambda x_outp_img=x_outp_img: self.lpmFinal(x_outp_img))

        self.dftFrame.pack_forget()
        self.frame.pack_forget()
        self.hpmFrame.pack_forget()
        self.bppmFrame.pack_forget()
        self.bspmFrame.pack_forget()
        self.lpmFrame.pack()

    def hpmCallBack(self, event, slider, x_outp_img):
        mx = len(x_outp_img)
        m2 = int((mx-1)/2)
        r = slider.get()
        x_outp_copy = np.copy(x_outp_img)

        circle = np.array([[(i-m2)**2+(j-m2)**2 for j in range(mx)] for i in range(mx)])
        circle = np.where(circle<=r**2)
        x_outp_copy[circle] = 1

        x_mag = image_utils.xMagCreate(x_outp_copy)

        self.hpmPanel.configure(image=x_mag)
        self.hpmPanel.image = x_mag

    def hpmFinal(self, x_outp_img):
        mx = len(x_outp_img)
        m2 = int((mx-1)/2)
        r = self.hpmSlider.get()
        circle = np.array([[(i-m2)**2+(j-m2)**2 for j in range(mx)] for i in range(mx)])
        circle = np.where(circle<=r**2)
        x_outp_img[circle] = 0

        self.maskFinal(x_outp_img)

    def hpmWrap(self):
        x_outp_img, x_mag = self.maskWrap()
        m = len(x_outp_img)

        self.hpmPanel.configure(image=x_mag)
        self.hpmPanel.image = x_mag

        self.hpmSlider.configure(to=m//2)
        self.hpmSlider.set(1)
        self.hpmSlider.pack(side=TOP)
        self.hpmSlider.bind("<ButtonRelease-1>", lambda event, slider=self.hpmSlider, x_outp_img=x_outp_img: self.hpmCallBack(event, slider, x_outp_img))

        self.hpmSubButton.configure(command=lambda x_outp_img=x_outp_img: self.hpmFinal(x_outp_img))

        self.dftFrame.pack_forget()
        self.frame.pack_forget()
        self.lpmFrame.pack_forget()
        self.bppmFrame.pack_forget()
        self.bspmFrame.pack_forget()
        self.hpmFrame.pack()

    def bppmCallBack(self, event, x_outp_img):
        mx = len(x_outp_img)
        m2 = int((mx-1)/2)
        r1 = self.bppmSliderLow.get()
        r2 = self.bppmSliderHigh.get()
        assert(r1<=r2)

        self.bppmSliderLow.configure(to=r2)
        self.bppmSliderHigh.configure(from_=r1)

        x_outp_copy = np.copy(x_outp_img)
        allVals = np.array([[(i-m2)**2+(j-m2)**2 for j in range(mx)] for i in range(mx)])
        nullVals = np.where(allVals<r1**2)
        x_outp_copy[nullVals] = 1
        nullVals = np.where(allVals>r2**2)
        x_outp_copy[nullVals] = 1

        x_mag = image_utils.xMagCreate(x_outp_copy)

        self.bppmPanel.configure(image=x_mag)
        self.bppmPanel.image = x_mag

    def bppmFinal(self,x_outp_img):
        mx = len(x_outp_img)
        m2 = int((mx-1)/2)
        r1 = self.bppmSliderLow.get()
        r2 = self.bppmSliderHigh.get()

        allVals = np.array([[(i-m2)**2+(j-m2)**2 for j in range(mx)] for i in range(mx)])
        nullVals = np.where(allVals<r1**2)
        x_outp_img[nullVals] = 0
        nullVals = np.where(allVals>r2**2)
        x_outp_img[nullVals] = 0

        self.maskFinal(x_outp_img)

    def bppmWrap(self):
        x_outp_img, x_mag = self.maskWrap()
        m = len(x_outp_img)

        self.bppmPanel.configure(image=x_mag)
        self.bppmPanel.image = x_mag

        self.bppmSliderHigh.configure(from_=1)
        self.bppmSliderHigh.configure(to=m//2)
        self.bppmSliderHigh.set(m//2)
        self.bppmSliderHigh.pack(side=TOP)
        self.bppmSliderHigh.bind("<ButtonRelease-1>", lambda event, x_outp_img=x_outp_img: self.bppmCallBack(event, x_outp_img))

        self.bppmSliderLow.configure(from_=1)
        self.bppmSliderLow.configure(to=m//2)
        self.bppmSliderLow.set(1)
        self.bppmSliderLow.pack(side=TOP)
        self.bppmSliderLow.bind("<ButtonRelease-1>", lambda event, x_outp_img=x_outp_img: self.bppmCallBack(event, x_outp_img))

        self.bppmSubButton.configure(command=lambda x_outp_img=x_outp_img: self.bppmFinal(x_outp_img))

        self.dftFrame.pack_forget()
        self.frame.pack_forget()
        self.lpmFrame.pack_forget()
        self.hpmFrame.pack_forget()
        self.bspmFrame.pack_forget()
        self.bppmFrame.pack()

    def bspmCallBack(self, event, x_outp_img):
        mx = len(x_outp_img)
        m2 = int((mx-1)/2)
        r1 = self.bspmSliderLow.get()
        r2 = self.bspmSliderHigh.get()
        assert(r1<=r2)

        self.bspmSliderLow.configure(to=r2)
        self.bspmSliderHigh.configure(from_=r1)

        x_outp_copy = np.copy(x_outp_img)
        allVals = np.array([[(i-m2)**2+(j-m2)**2 for j in range(mx)] for i in range(mx)])
        nullVals = np.where((allVals-r1**2)*(allVals-r2**2)<0)
        x_outp_copy[nullVals] = 1

        x_mag = image_utils.xMagCreate(x_outp_copy)

        self.bspmPanel.configure(image=x_mag)
        self.bspmPanel.image = x_mag

    def bspmFinal(self,x_outp_img):
        mx = len(x_outp_img)
        m2 = int((mx-1)/2)
        r1 = self.bspmSliderLow.get()
        r2 = self.bspmSliderHigh.get()

        allVals = np.array([[(i-m2)**2+(j-m2)**2 for j in range(mx)] for i in range(mx)])
        nullVals = np.where((allVals-r1**2)*(allVals-r2**2)<0)
        x_outp_img[nullVals] = 0

        self.maskFinal(x_outp_img)

    def bspmWrap(self):
        x_outp_img, x_mag = self.maskWrap()
        m = len(x_outp_img)

        self.bspmPanel.configure(image=x_mag)
        self.bspmPanel.image = x_mag

        self.bspmSliderHigh.configure(from_=1)
        self.bspmSliderHigh.configure(to=m//2)
        self.bspmSliderHigh.set(m//2)
        self.bspmSliderHigh.pack(side=TOP)
        self.bspmSliderHigh.bind("<ButtonRelease-1>", lambda event, x_outp_img=x_outp_img: self.bspmCallBack(event, x_outp_img))

        self.bspmSliderLow.configure(from_=1)
        self.bspmSliderLow.configure(to=m//2)
        self.bspmSliderLow.set(1)
        self.bspmSliderLow.pack(side=TOP)
        self.bspmSliderLow.bind("<ButtonRelease-1>", lambda event, x_outp_img=x_outp_img: self.bspmCallBack(event, x_outp_img))

        self.bspmSubButton.configure(command=lambda x_outp_img=x_outp_img: self.bspmFinal(x_outp_img))

        self.dftFrame.pack_forget()
        self.frame.pack_forget()
        self.lpmFrame.pack_forget()
        self.hpmFrame.pack_forget()
        self.bppmFrame.pack_forget()
        self.bspmFrame.pack()

    def fullInverseWrap(self):
        img = np.array(self.img)
        self.prevImg = self.img
        imgNew = filtering.inverseFilter2D(img, self.filter_kernel)
        self.displayImg(imgNew)

    def truncatedInverseWrap(self): 
        img = np.array(self.img)
        self.prevImg = self.img
        self.truncatedInverseFilter = filtering.getTruncatedInverseFilter2D(img, self.filter_kernel)
        self.truncatedInverseFrame.pack()
        self.truncatedInverseSlider.set(1.0)
        self.truncatedInverseSlider.bind("<ButtonRelease-1>", lambda event, img=img: self.truncatedInverseCallback(event, img))

    def approximateWeinerWrap(self): 
        img = np.array(self.img)
        self.prevImg = self.img
        self.weinerFilter = filtering.getApproximateWeinerFilter2D(img, self.filter_kernel)
        self.weinerFrame.pack()
        self.weinerSlider.set(1.0)
        self.weinerSlider.bind("<ButtonRelease-1>", lambda event, img=img: self.weinerCallback(event, img))

    def contrainedLSWrap(self): # TODO: implement 
        img = np.array(self.img)
        self.prevImg = self.img
        self.clsFilter = filtering.getConstrainedLSFilter2D(img, self.filter_kernel)
        self.clsFrame.pack()
        self.clsSlider.set(1.0)
        self.clsSlider.bind("<ButtonRelease-1>", lambda event, img=img: self.clsCallback(event, img))

root = Tk()
ipGUI(root)
root.mainloop()
