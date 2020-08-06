# Copyright (c) 2019-2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import sys
import numpy as np

import pyeddl.eddl as eddl
import pyecvl.ecvl as ecvl
from pyeddl.tensor import Tensor

import models

try:
    import openslide
    no_openslide = False
except:
    no_openslide=True


class tissue_detector:
    def __init__(self, th=0.5, model_fn=None, gpu=True):
        self.model = None   # Model used for prediction
        self.th = th        # Threshold used to make the tissue mask binary if the model returns prob
        self.gpu = gpu
        #bs: batch size if a tensorflow ann model is used. Change the default value 
        #    to a lower one if the GPU goes out of memory
        #self.bs = 262144
        self.bs = 2**24

        if model_fn:
            self.load_model(model_fn)
    
    def __get_binary_mask(self, prob_T_l, s, th):
        """
        @prob_T_L: list of probability tensors resulting from model predictions.
                  Once each tensor is binarized, a single array is created stacking them
                  and the image mask is created reshaping the array
        
        @mask: The resulting tissue mask
        """
        
        mask_np_l = []
        for prob_T in prob_T_l:
            output_np = prob_T.getdata()
            pred_np = np.zeros(output_np.shape[0])
            pred_np[output_np[:, 1]>th] = 1
            mask_values = pred_np
            mask_np_l.append(mask_values)

        mask_values = np.vstack(mask_np_l)
        mask = mask_values.reshape((s[0], s[1]))

        return mask

    def load_model(self, model_weights):
        ## Load the ANN tissue detector model implemented by using pyeddl
        ## Create ANN topology (Same used during training phase)
        net = models.tissue_detector_DNN()
        eddl.build(
            net,
            eddl.rmsprop(0.00001),
            ["soft_cross_entropy"],
            ["categorical_accuracy"],
            eddl.CS_GPU() if self.gpu else eddl.CS_CPU()
        )
        # Load weights
        eddl.load(net, model_weights, "bin")
        
        self.model = net
    
    def get_tissue_mask(self, np_img, channel_first=True):
        """
        @np_img: numpy array of a PIL image (x,y,4) if alpha channel is present 
                 or (x, y, 3) if alpha channel not present
        @channel_first: boolean to specify if the image is channel first (3,x,y) 
                        or channel last (x,y,3) (3 is replaced by 4 if alpha is present)
        
        returns a (x,y) numpy array with binary values (0:bg, 1:tissue)
        """

        if channel_first:
            np_img = np_img.transpose((1,2,0)) # Convert to channel last
            
        s = np_img.shape
        n_px = s[0] * s[1]
        np_img = np_img[:,:,:3].reshape(n_px, 3)
        
        if not self.model:
            print ("Cannot make predictions without a model. Please, load one!")
            return None
        
        ## From numpy array to eddl tensor to predict 
        t_eval = Tensor.fromarray(np_img)

        output_l = eddl.predict(self.model, [t_eval]) # Prediction.. get probabilities
        msk_pred = self.__get_binary_mask(output_l, s, self.th) ## Get the actual mask (binarization)

        return msk_pred

    def get_mask_tissue_from_slide(self, slide_fn, level=2, use_openslide=False):
        
        if use_openslide and not no_openslide:
            # Using open slide
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

            msk_pred = self.get_tissue_mask(np_img, channel_first=False)
        else:
            ## Using ECVL
            levels = ecvl.OpenSlideGetLevels(slide_fn)
            dims = [0, 0] + levels[level]
            img = ecvl.OpenSlideRead(slide_fn, level, dims)
            t = ecvl.ImageToTensor(img)
            np_img = t.getdata() # Tensor to numpy array (ecvl read images with channel first)
            
            msk_pred = self.get_tissue_mask(np_img, channel_first=True)

        return msk_pred
