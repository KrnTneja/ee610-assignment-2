import cv2
import numpy as np

from PIL import Image
from PIL import ImageTk

import fft

def findHistogram(img):
    # Histogram Equalization
    # img is assumed in hsv
    m,n = img.shape[:2]
    hist = np.zeros(256)
    for i in range(m):
        for j in range(n):
            hist[img[i][j][2]] += 1
    return hist

def histEqlFunc(img):
    # img in RGB, return imgNew RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m,n = img.shape[:2]
    hist = findHistogram(img)
    l = 256
    transform = (l-1)*np.array([sum(hist[:i+1]) for i in range(l)])/(m*n)
    transform = np.array(np.round(transform), dtype=np.uint8)
    imgNew = np.array([[[img[i][j][0], img[i][j][1], transform[img[i][j][2]]] for j in range(n)] for i in range(m)])
    imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)
    return imgNew

def gammaCorFunc(img, c=1.0, gamma=1.0):
    # img in RGB, return imgNew RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m,n = img.shape[:2]
    transform = [min(c*(r**gamma), 255) for r in range(256)] # cap values to 255
    transform = np.array(np.round(transform), dtype=np.uint8)
    imgNew = np.array([[[img[i][j][0], img[i][j][1], transform[img[i][j][2]]] for j in range(n)] for i in range(m)])
    imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)
    return imgNew

def logTranFunc(img, c=1.0):
    # Log Transform
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    m,n = img.shape[:2]
    transform = [min(c*np.log10(1+r),255) for r in range(256)]
    transform = np.array(np.round(transform), dtype=np.uint8)
    imgNew = np.array([[[img[i][j][0], img[i][j][1], transform[img[i][j][2]]] for j in range(n)] for i in range(m)])
    imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)
    return imgNew

def inner_product(a, b):
    # Inner product
    assert a.size == b.size
    return sum(np.multiply(a.flatten(), b.flatten()))

def correlate(img, filt, orig_weight=0.0, filt_weight=1.0):
    # Correlation function
    assert len(img.shape) == 2
    assert filt.shape[0] == filt.shape[1] and filt.shape[0] % 2 == 1 and len(filt.shape) == 2

    filt_size = filt.shape[0]
    padding_size = filt_size//2
    img_h, img_w = img.shape

    inp = np.zeros((img_h+2*padding_size, img_w+2*padding_size), dtype=np.float32)
    inp[padding_size:-padding_size, padding_size:-padding_size] = np.float32(img)
    filt_arr = np.float32(filt)/(1.0 if np.sum(filt) == 0 else np.sum(filt))

    output = orig_weight*img + filt_weight*np.array([[inner_product(inp[i:i+filt_size, j:j+filt_size], filt_arr)
                                                      for j in range(img_w)] for i in range(img_h)])
    output = np.clip(output, 0, 255)
    return np.int8(output)

def sharpFunc(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    filt = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    imgNew = img.copy()
    imgNew[:, :, 2] = correlate(imgNew[:, :, 2], filt, orig_weight=1.0, filt_weight=0.3)
    imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)

    return imgNew

def boxBlurFunc(event, img, filterSize):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    filt = np.ones((filterSize, filterSize))
    imgNew = img.copy()
    imgNew[:, :, 2] = correlate(imgNew[:, :, 2], filt)
    imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)
    return imgNew

def gaussianBlurFunc(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    filt = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
    imgNew = img.copy()
    imgNew[:, :, 2] = correlate(imgNew[:, :, 2], filt)
    imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)

    return imgNew

def medianBlurFunc(img):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    filt_size = 3
    padding_size = filt_size//2
    img_h, img_w = img.shape[:2]

    inp = np.zeros((img_h+2*padding_size, img_w+2*padding_size), dtype=np.float32)
    inp[padding_size:-padding_size, padding_size:-padding_size] = np.float32(img[:, :, 2])

    median = lambda x: np.sort(x.flatten())[x.size//2]
    output = np.array([[median(inp[i:i+filt_size, j:j+filt_size])
                        for j in range(img_w)] for i in range(img_h)])
    output = (output-np.min(output))/(np.max(output)-np.min(output))*255
    imgNew = img.copy()
    imgNew[:, :, 2] = output
    imgNew = cv2.cvtColor(imgNew, cv2.COLOR_HSV2RGB)
    return imgNew

def cartoonifyFunc(img):
    # img in RGB, newImg in RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    imgNew = center[label.flatten()]
    imgNew = imgNew.reshape((img.shape))
    imgNew = cv2.cvtColor(imgNew, cv2.COLOR_BGR2RGB)
    return imgNew

def freqMask(img, mask):
    # img in RGB, res in RGB
    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    x = img[:,:,2]
    m, n = x.shape
    x_outp_img = fft.fft2d_img(x)
    res_fft = np.multiply(x_outp_img, mask)/255
    res = np.absolute(fft.ifft2d_img(res_fft))
    res = np.array(res*255/np.max(res),dtype=np.uint8)
    res = np.array([[[img[i][j][0], img[i][j][1], res[i][j]] for j in range(m)] for i in range(n)])
    res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
    return res

def xMagCreate(x_outp_img):
    m = len(x_outp_img)
    x_mag = np.log10(np.absolute(x_outp_img))*255/np.log10(m*m*255)
    x_mag = Image.fromarray(x_mag)
    x_mag = x_mag.resize((min(257,m), min(257,m)), Image.ANTIALIAS)
    x_mag = ImageTk.PhotoImage(x_mag)
    return x_mag
