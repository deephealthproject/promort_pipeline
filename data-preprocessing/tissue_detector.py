import os
import sys
import pickle 
import numpy as np
import openslide

try:
    import tensorflow as tf
    from tensorflow import keras
    tf_gpu = True
    if tf.__version__ < "2.2.0":
        print ("Tensorflow version is %f. It must be >= 2.2")
        tf_gpu = False
except:
    tf_gpu = False


class tissue_detector:
    def __init__(self, th=0.5, model_fn=None):
        self.model = None   # Model used for prediction
        self.th = th        # Threshold used to make the tissue mask binary if the model returns prob
        #bs: batch size if a tensorflow ann model is used. Change the default value 
        #    to a lower one if the GPU goes out of memory
        self.bs = 262144

        if model_fn:
            self.load_model(model_fn)
    
    def load_model(self, model_fn):
        #Check if model filename is a dir (tensorflow model) or a file (sklearn model)
        self.ann_model = False
        if os.path.isdir(model_fn) and tf_gpu:
            self.model = keras.models.load_model(model_fn)
            self.ann_model = True
        elif os.path.isdir(model_fn) and not tf_gpu:
            print ("Cannot load a Keras model. Gpu not enabled.")
        else:
            self.model = pickle.load(open(model_fn, 'rb'))
    
    def get_tissue_mask(self, np_img):
        """
        @np_img: numpy array of a PIL image (x,y,4) if alpha channel is present 

        returns a (x,y) numpy array with binary values (0:bg, 1:tissue)
        """

        n_px = np_img.shape[0] * np_img.shape[1]
        x = np_img[:,:,:3].reshape(n_px, 3)
        
        if not self.model:
            print ("Cannot make predictions without a model. Please, load one!")
            return None
            
        if self.ann_model:
            pred = self.model.predict(x, batch_size=self.bs)
        else:
            pred = self.model.predict(x)

        msk_pred = pred.reshape(np_img.shape[0], np_img.shape[1])
        msk_pred[msk_pred < self.th] = 0
        msk_pred[msk_pred > self.th] = 1

        return msk_pred

    def get_mask_tissue_from_slide(self, slide_fn, level=2):

        # open slide
        slide = openslide.open_slide(slide_fn)
        dims = slide.level_dimensions
        ds = [int(i) for i in slide.level_downsamples]

        x0 = 0
        y0 = 0
        x1 = dims[0][0]
        y1 = dims[0][1]

        delta_x = x1 - x0
        delta_y = y1 - y0

        pil_img = slide.read_region(location=(x0, y0), level=level, size=(delta_x // ds[level], delta_y // ds[level]))
        np_img = np.array(pil_img)

        msk_pred = self.get_tissue_mask(np_img)

        return msk_pred
