import warnings
warnings.filterwarnings("ignore")
from flask import Flask
from flask import request, jsonify
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import base64

# import matplotlib.pyplot as plt
from scipy.signal import medfilt
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage import feature
from skimage.filters import roberts, sobel

import tensorflow as tf
# from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
# from keras.models import Sequential, Model
# from keras.layers import Dense, Dropout, Activation, Flatten, Input, GlobalMaxPooling2D
# from keras.layers.normalization import BatchNormalization
from keras.models import model_from_json
from keras.models import load_model
INPUT_SIZE = 512


urllink = "https://storage.googleapis.com/kaggle-data-sets/397442/763854/bundle/archive.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1572287891&Signature=DLXGz%2FgxIxuBqPHX7GXq%2BT2qjtgTiU9%2FiyJ9RuOnxZRiss%2BAUB7xmuTzEH%2FtqanETKF9PnWTif9nSt6jXeh%2FNhS%2B2ni0%2Byc%2BXlCb%2BnX2h319wVpJk9tevCdAh%2F59uo5IjtuWY1E0Vd62UWL7ks559nGeVis%2B%2BWMGpOfJiF8imOQlCUZMpZJ6S4g7oZ5n%2FRCREC72BX7vHEkP0TH6fxO3MMOssdyJKSwpoqH%2BKIQPe7UchkLy%2FwOhQVOVcUCR0qoKuIBe8sV68pcRh5T3SAwDV5cZ0EEXxJUe4ZB4yEVJb0pNzhxnJpT0od1mSxQ2CpEUmJdBa7jhA862xyP1WdykDw%3D%3D&response-content-disposition=attachment%3B+filename%3Dbest-weights-inceptionresnetv2.zip"
weights_path =  tf.keras.utils.get_file(fname = 'best_weights_inceptionresnetv2.hdf5', origin=urllink, extract=True)
def model_load_weight():
        # input_tensor = Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
        # base_model = InceptionResNetV2(include_top=False, weights=None, input_shape=(INPUT_SIZE, INPUT_SIZE, 3),backend=tf.keras.backend, layers=tf.keras.layers, models=tf.keras.models, utils=tf.keras.utils)
        # bn = BatchNormalization()(input_tensor)
        # x = base_model(bn)
        # x = GlobalMaxPooling2D()(x)
        # x = Dense(1024, activation='relu')(x)
        # x = Dropout(0.25)(x)
        # output_tensor = Dense(16, activation='relu', name="pred_point")(x)
        # model = Model(inputs=input_tensor, outputs=output_tensor)

        # Load model from file json
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # load weights into new model
        model.load_weights(weights_path)
        print("Loaded model from disk")
        # load  model
        # model = load_model('model_and_best_weights.h5')
        model.summary()
        model._make_predict_function()
        return model
# model = model_load_weight()


def denoiseFFT(im, keep_fraction = 0.30):
    from scipy import fftpack
    im_fft = fftpack.fft2(im)
    im_fft2 = im_fft.copy()

    # Set r and c to be the number of rows and columns of the array.
    r, c = im_fft2.shape

    # Set to zero all rows with indices between r*keep_fraction and
    # r*(1-keep_fraction):
    im_fft2[int(r*keep_fraction):int(r*(1-keep_fraction))] = 0

    # Similarly with the columns:
    im_fft2[:, int(c*keep_fraction):int(c*(1-keep_fraction))] = 0
    
    im_new = fftpack.ifft2(im_fft2).real
    
    return im_new

# FOR POINT 2 AND 5
def largest_object(img):
    label_image =  label(img)
    IndexList = list(np.unique(label_image))[1:]
    areas = [region.area for region in regionprops(label_image)]
    idx = IndexList[np.argmax(areas)]
    MaxAreaBW = label_image==idx
    return MaxAreaBW, np.max(areas)

def get_skeleton(img):
    skeleton = skeletonize(img)
    label_image =  label(skeleton)
    IndexList = list(np.unique(label_image))[1:]
    areas = [region.area for region in regionprops(label_image)]
    idx = IndexList[np.argmax(areas)]
    MaxAreaBW = label_image==idx
    return skeleton, np.max(areas)

def largest_object_and_skeleton(img):
    label_image =  label(img)
    IndexList = list(np.unique(label_image))[1:]
    areas = [region.area for region in regionprops(label_image)]
    idx = IndexList[np.argmax(areas)]
    MaxAreaBW = label_image==idx
    # skealeton
    skeleton, ar = get_skeleton(MaxAreaBW)
    return skeleton, ar

# FOR POINT 4 AND 7
def print_grad_left(ml):
    nr, nc = ml.shape
    step = 2
    flag = False
    sc, sr = 0,0
    for i in range(nc-1,-1,-step):
        bf = ml[:,i]
        af = ml[:,np.max(i-step,0)]
        ybf = np.argmax(bf)
        yaf = np.argmax(af)
        grad = max(0, ybf-yaf)
#         print(i, grad)
        if (not flag ) and (grad >=2):
            sc, sr = i, ybf
            flag=True
            break
    return sc, sr

def find_nearest_left(ml, pyi, pxi):
    step = 2
    nr, nc = ml.shape
    py, px = int(np.round(pyi)), int(np.round(pxi))
    if ml[px, py] > 0: return pyi, pxi
    sc, sr = py, px
    ry, rx = py, px
    while (rx > 1) and (ry < nc-1): # run upper right
        rx-=1
        ry+=1
        if (ml[rx, ry] > 0): 
            sc, sr = ry, rx
            break
        try:
            if (ml[rx-1, ry] > 0) or (ml[rx, ry+1] > 0): 
                sc, sr = ry, rx
                break
        except: continue
    ry, rx = py, px
    while (ry > 1) and (rx < nr-1):# run lower left
        ry -= 1
        rx += 1
        if (ml[rx, ry] > 0): 
            sc, sr = ry, rx
            break
        try:
            if (ml[rx+1, ry] > 0) or (ml[rx, ry-1] > 0): 
                sc, sr = ry, rx
                break
        except: continue
            
    flag = False
    sc1, sr1 = sc, sr 
    for i in range(sc,-1,-step):
        bf = ml[:,i]
        af = ml[:,np.max(i-step,0)]
        ybf = np.argmax(bf)
        yaf = np.argmax(af)
        grad = max(0, ybf-yaf)
#         print(i, grad)
        if (not flag ) and (grad >=4):
            sc1, sr1 = i, ybf
            break
    return sc1, sr1
# FOR POINT 4 AND 7
def print_grad_right(ml):
    nr, nc = ml.shape
    step = 2
    flag = False
    sc, sr = 0,0
    for i in range(0,nc-step,step):
        bf = ml[:,i]
        af = ml[:,np.max(i+step,0)]
        ybf = np.argmax(bf)
        yaf = np.argmax(af)
        grad = max(0, ybf-yaf)
#         print(i, grad)
        if (not flag ) and (grad >=2):
            sc, sr = i, ybf
            flag=True
            break
    return sc, sr

def find_nearest_right(ml, pyi, pxi):
    step = 2
    nr, nc = ml.shape
    py, px = int(np.round(pyi)), int(np.round(pxi))
    if ml[px, py] > 0: return pyi, pxi
    sc, sr = py, px
    ry, rx = py, px
    while (rx > 1) and (ry >1): # run upper left
        rx-=1
        ry-=1
        if (ml[rx, ry] > 0): 
            sc, sr = ry, rx
            break
        try:
            if (ml[rx-1, ry] > 0) or (ml[rx, ry-1] > 0): 
                sc, sr = ry, rx
                break
        except: continue
    ry, rx = py, px
    while (ry > nc-1) and (rx < nr-1):# run lower right
        ry += 1
        rx += 1
        if (ml[rx, ry] > 0): 
            sc, sr = ry, rx
            break
        try:
            if (ml[rx+1, ry] > 0) or (ml[rx, ry+1] > 0): 
                sc, sr = ry, rx
                break
        except: continue
            
    flag = False
    sc1, sr1 = sc, sr 
    for i in range(0,nc-step,step):
        bf = ml[:,i]
        af = ml[:,np.max(i+step,0)]
        ybf = np.argmax(bf)
        yaf = np.argmax(af)
        grad = max(0, ybf-yaf)
#         print(i, grad)
        if (not flag ) and (grad >=4):
            sc1, sr1 = i, ybf
            flag=True
            break
    return sc1, sr1
def post_process_00(img, p):
    y, x = int(np.ceil(p[0])), int(np.ceil(p[1]))
    right = 16
    left = 10
    step = 2
    h = 15
    l = 10
    # yl, xl = pt[0] - (y-left) - 5, pt[1] - (x-h) - 5
    yp, xp = p[0] - (y-left) - 5, p[1] - (x-h) - 5
    arr1 =  np.asarray(medfilt(img[x-h:x+l, y-left: y+right],7), np.uint8)
    edge_roberts = roberts(arr1)
    edge_sobel = sobel(arr1)
    edge_canny = cv2.Canny(arr1,50,100,apertureSize = 3)
    cond = edge_sobel
    
    cond = np.asarray(cond > np.percentile(cond, 88), dtype=np.uint8)
    try: cond, ar = largest_object_and_skeleton(cond[5:-5,5:-5])
    except :cond, ar = cond[5:-5,5:-5], 0
#     print("area", ar)
    cond = np.asarray(cond, dtype=np.uint8)
    pxs, pys = np.where(cond > 0)
    para_a = np.polyfit(pxs, pys, 2)[0]
    sc, sr = yp, xp
    if para_a <= 0:
        sc = np.where(np.sum(cond,axis=0)>0)[0][-1]
        sr = np.where(cond[:,sc]> 0)[0][-1]
    delta_col = sc - yp
    delta_row = sr - xp
#     print(delta_col, delta_row)
    yr, xr = p[0] + delta_col/2, p[1] + delta_row/2
    return yr, xr

def post_process_01(img, p):
    y, x = int(np.ceil(p[0])), int(np.ceil(p[1]))
    right = 10
    left = 16
    step = 2
    h = 15
    l = 10
    # yl, xl = pt[0] - (y-left) - 5, pt[1] - (x-h) - 5
    yp, xp = p[0] - (y-left) - 5, p[1] - (x-h) - 5
    arr1 =  np.asarray(medfilt(img[x-h:x+l, y-left: y+right],7), np.uint8)
    edge_roberts = roberts(arr1)
    edge_sobel = sobel(arr1)
    edge_canny = cv2.Canny(arr1,50,100,apertureSize = 3)
    cond = edge_sobel
    
    cond = np.asarray(cond > np.percentile(cond, 88), dtype=np.uint8)
    try: cond, ar = largest_object_and_skeleton(cond[5:-5,5:-5])
    except :cond, ar = cond[5:-5,5:-5], 0
#     print("area", ar)
    cond = np.asarray(cond, dtype=np.uint8)
    pxs, pys = np.where(cond > 0)
    para_a = np.polyfit(pxs, pys, 2)[0]
    sc, sr = yp, xp
    if para_a >= 0:
        sc = np.where(np.sum(cond,axis=0)>0)[0][0]
        sr = np.where(cond[:,sc]> 0)[0][0]
    delta_col = sc - yp
    delta_row = sr - xp
#     print(delta_col, delta_row)
    yr, xr = p[0] + delta_col/2, p[1] + delta_row/2
    return yr, xr

def post_process_02(img, p):

    y, x = int(np.ceil(p[0])), int(np.ceil(p[1]))

    right = 32
    left = 32
    step = 2
    h = 18
    l = 10
    # yl, xl = pt[0] - (y-left) - 5, pt[1] - (x-h) - 5
    yp, xp = p[0] - (y-left) - 5, p[1] - (x-h) - 5
    arr1 =  np.asarray(medfilt(img[x-h:x+l, y-left: y+right],7), np.uint8)
    edge_roberts = roberts(arr1)
    edge_sobel = sobel(arr1)
    edge_canny = cv2.Canny(arr1,50,40,apertureSize = 3)
    cond = edge_sobel
    
    cond = np.asarray(cond > np.percentile(cond, 85), dtype=np.uint8)
    try: cond, ar = largest_object_and_skeleton(cond[5:-5,5:-5])
    except :cond, ar = cond[5:-5,5:-5], 0
    cond = np.asarray(cond, dtype=np.uint8)
    c_idx = yp
    r_idx = xp
    if ar > 0:
        try:
            c_idx = int(np.round(np.mean(np.where(np.sum(cond,axis=0)>0)[0][0])))
            r_idx = int(np.round(np.mean(np.where(cond[:, c_idx])[0])))
#             print(c_idx, r_idx)
        except:pass
    delta_col = c_idx - yp
    delta_row = r_idx - xp
    dg = np.sqrt(delta_col**2+delta_row**2)
#     print("Distance go: ", dg)
    if np.abs(delta_row) < 8 and ar > 18:
        pass
    else: 
        if ar > 32:
            pass
        else:
            delta_col, delta_row = 0.0, 0.0
    yr, xr = p[0] + delta_col, p[1] + delta_row
    return yr, xr

def post_process_03(img, p):
    y, x = int(np.ceil(p[0])), int(np.ceil(p[1]))
    length = 10
    right = 8
    step = 3
    h = 5
    arr1 = medfilt(img[x-h+1:x+h-3+1, y-length: y+right],5)
    arr2 = medfilt(img[x-h:x+h-3, y-length-step: y+right-step],5)
    grad = medfilt(arr1 - arr2,3)
    
    cond = np.asarray(grad < np.percentile(grad, 12), dtype=np.uint8)
    
    x_del = np.where(np.max(cond,axis=0)==np.amax(np.max(cond,axis=0)))[0][-1]
    y_del = np.mean(np.where(cond[:,int(x_del)]==np.amax(cond[:,int(x_del)])))
    y = p[0] + x_del - 10
    x = p[1] + y_del - 4
    return (y, x)
def post_process_04(img, p):
    
    y, x = int(np.ceil(p[0])), int(np.ceil(p[1]))
    right = 16
    left = 16
    step = 2
    h = 15
    l = 10
    # yl, xl = pt[0] - (y-left) - 5, pt[1] - (x-h) - 5
    yp, xp = p[0] - (y-left) - 5, p[1] - (x-h) - 5
    arr1 =  np.asarray(medfilt(img[x-h:x+l, y-left: y+right],7), np.uint8)
    edge_roberts = roberts(arr1)
    edge_sobel = sobel(arr1)
    edge_canny = cv2.Canny(arr1,50,100,apertureSize = 3)
    cond = edge_sobel
    
    cond = np.asarray(cond > np.percentile(cond, 88), dtype=np.uint8)
    try: cond, ar = largest_object_and_skeleton(cond[5:-5,5:-5])
    except :cond, ar = cond[5:-5,5:-5], 0

    cond = np.asarray(cond, dtype=np.uint8)
    sc, sr = print_grad_left(cond)
    
    delta_col = sc - yp
    delta_row = sr - xp
    dg = np.sqrt(delta_col**2+delta_row**2)
    
    if dg > 6:
        sc, sr = find_nearest_left(cond, yp, xp)
    delta_col = sc - yp
    delta_row = sr - xp
    return p[0]+delta_col, p[1]+delta_row

def post_process_05(img, p):
    
    y, x = int(np.ceil(p[0])), int(np.ceil(p[1]))

    right = 32
    left = 32
    step = 2
    h = 18
    l = 10
    # yl, xl = pt[0] - (y-left) - 5, pt[1] - (x-h) - 5
    yp, xp = p[0] - (y-left) - 5, p[1] - (x-h) - 5
    arr1 =  np.asarray(medfilt(img[x-h:x+l, y-left: y+right],7), np.uint8)
    edge_roberts = roberts(arr1)
    edge_sobel = sobel(arr1)
    edge_canny = cv2.Canny(arr1,50,40,apertureSize = 3)
    cond = edge_sobel
    
    cond = np.asarray(cond > np.percentile(cond, 85), dtype=np.uint8)
    try: cond, ar = largest_object_and_skeleton(cond[5:-5,5:-5])
    except :cond, ar = cond[5:-5,5:-5], 0
    cond = np.asarray(cond, dtype=np.uint8)
    c_idx = yp
    r_idx = xp
    if ar > 0:
        try:
            c_idx = int(np.round(np.mean(np.where(np.sum(cond,axis=0)>0)[0][-1])))
            r_idx = int(np.round(np.mean(np.where(cond[:, c_idx])[0])))
        except:pass
    delta_col = c_idx - yp
    delta_row = r_idx - xp
    dg = np.sqrt(delta_col**2+delta_row**2)
    if np.abs(delta_row) < 8 and ar > 18:
        pass
    else: 
        if ar > 32:
            pass
        else:
            delta_col, delta_row = 0.0, 0.0
    yr, xr = p[0] + delta_col, p[1] + delta_row
    return yr, xr

def post_process_06(img, p):
    y, x = int(np.ceil(p[0])), int(np.ceil(p[1]))
    length = 8
    left = 4
    step = 3
    h = 8
    arr1 = medfilt(img[x-h+1:x+h-7+1, y-left: y+length],5)
    arr2 = medfilt(img[x-h:x+h-7, y-left-step: y+length-step],5)
    grad = medfilt(arr2 - arr1,3)
    
    cond = np.asarray(grad < np.percentile(grad, 12), dtype=np.uint8)
    cond = cv2.morphologyEx(cond, cv2.MORPH_OPEN, np.ones((2,1), np.uint8))
    x_del = np.where(np.max(cond,axis=0)==np.amax(np.max(cond,axis=0)))[0][0]
    y_del = np.mean(np.where(cond[:,int(x_del)]==np.amax(cond[:,int(x_del)])))
    y = p[0] + x_del - 4
    x = p[1] + y_del - 6
    
    return (y,x)

def post_process_07(img, p):
    
    y, x = int(np.ceil(p[0])), int(np.ceil(p[1]))
    right = 16
    left = 16
    step = 2
    h = 15
    l = 10
    # yl, xl = pt[0] - (y-left) - 5, pt[1] - (x-h) - 5
    yp, xp = p[0] - (y-left) - 5, p[1] - (x-h) - 5
    arr1 =  np.asarray(medfilt(img[x-h:x+l, y-left: y+right],7), np.uint8)
    edge_roberts = roberts(arr1)
    edge_sobel = sobel(arr1)
    edge_canny = cv2.Canny(arr1,50,100,apertureSize = 3)
    cond = edge_sobel
    
    cond = np.asarray(cond > np.percentile(cond, 88), dtype=np.uint8)
    try: cond, ar = largest_object_and_skeleton(cond[5:-5,5:-5])
    except :cond, ar = cond[5:-5,5:-5], 0
    cond = np.asarray(cond, dtype=np.uint8)
    sc, sr = print_grad_right(cond)
    
    delta_col = sc - yp
    delta_row = sr - xp
    dg = np.sqrt(delta_col**2+delta_row**2)
    if dg > 6:
        sc, sr = find_nearest_right(cond, yp, xp)
    delta_col = sc - yp
    delta_row = sr - xp
    return p[0]+delta_col, p[1]+delta_row



class HipProcessing(object):
    def __init__(self):
        self.target_shape = (INPUT_SIZE, INPUT_SIZE,3)
        self.model = model_load_weight()

    

    def modify_predict(self, image, lb):
        new_prd =[]
        img = denoiseFFT(image)
        yp = list(zip(*[iter(lb)]*2))

        for ii, lm in enumerate(yp):
            y, x = yp[ii][0],yp[ii][1]
            if ii == 0:
                y, x = post_process_00(img, [y,x])
            if ii == 1:
                y, x = post_process_01(img, [y,x])
            if ii == 2:
                y, x = post_process_02(img, [y,x])
            if ii == 3:
                y, x = post_process_03(img, [y,x])
            if ii == 4:
                y, x = post_process_04(img, [y,x])
            if ii == 5:
                y, x = post_process_05(img, [y,x])
            if ii == 6:
                y, x = post_process_06(img, [y,x])
            if ii == 7:
                y, x = post_process_07(img, [y,x])
            new_prd.extend([y,x])
        return new_prd
    
    def cropFunc(self, img):
        '''
        Cropping image to square have size a x a 
        with a = min(Hight, Width) from the centre of image
        '''
        h, w = img.shape
        a = min(h, w)
        h_h, w_h, a_h = int(h/2), int(w/2), int(a/2)
        h0, w0 = h_h-a_h, w_h-a_h
        new_img = img[h0:h0+a, w0:w0+a]
        return new_img, h0, w0

    def enhanceFunc(self, img):
        kernel = np.array([[-1,-1,-1], 
                       [-1, 9,-1],
                       [-1,-1,-1]])

        blur1 = cv2.GaussianBlur(img ,(5,5),cv2.BORDER_DEFAULT)
        equ1 = cv2.equalizeHist(blur1) 
        sharpened1 = cv2.filter2D(equ1, -1, kernel)
        return sharpened1
        
    def hip_points_detection(self, image):
        # Get hight and width
        inH, inW = image.shape
        # Crop image to square image
        img_crop, h0, w0 = self.cropFunc(image)
        # enhance contrast
        img_enh = np.asarray(self.enhanceFunc(img_crop), np.float32)
        # increase dim
        img_rep = np.repeat(img_enh[:, :, np.newaxis], 3, axis=2)
        # square shape
        orig_shape = img_rep.shape[0]
        # reshape to input model
        img_reshape = cv2.resize(img_rep, self.target_shape[:2])
        # ratio scale
        rat = self.target_shape[0] / orig_shape
        # scale to range (-1, 1)
        img_scale = img_reshape / (127.5) - 1
        # predict output
        pred_points = self.model.predict(img_scale[np.newaxis,:,:,:])
        pred_points = self.modify_predict(img_reshape[:,:,0], pred_points[0])

        # rescale to original
        pp_rescale = np.asarray(pred_points) / rat
        # plus to crop
        pp_recrop = pp_rescale.reshape(8,2) + np.array([w0, h0 ])
        
        output = pp_recrop.flatten()

        return output