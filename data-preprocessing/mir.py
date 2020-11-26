import multiresolutionimageinterface as mrii
import os

class MRimage() :
    # pl = level of dezoom in the multiresolution pyramid (0 = full detail)
    def __init__(self, pl = 0) :
        self.reader = mrii.MultiResolutionImageReader()
        self.maskReader = mrii.MultiResolutionImageReader()
        self.tissueReader = mrii.MultiResolutionImageReader()
        self.pyr_level = pl
        self.annotation_list = mrii.AnnotationList()
        self.xml_repository = mrii.XmlRepository(self.annotation_list)
    # open TIFF multiresolution image
    def open(self, filename) :
        self.pyram = self.reader.open(filename)
        self.ds = self.pyram.getLevelDownsample(self.pyr_level)
    # get image sizes
    def getSizes(self) :
        return self.pyram.getDimensions()
    # get downsample factor (1.0 = full size)
    def getDS(self) :
        return self.ds
    # open TIFF mask, created with XML2TIFF
    def openMask(self, filename) :
        self.mask = self.maskReader.open(filename)
        self.ds_mask = self.mask.getLevelDownsample(self.pyr_level)
    # open tissue mask
    def openTissue(self, filename) :
        self.tissue = self.tissueReader.open(filename)
        self.ds_tissue = self.tissue.getLevelDownsample(self.pyr_level)
    # get RGB Patch
    def getRGBPatch(self, x, y, dx, dy) :
        ds = self.ds
        rgb_patch = self.pyram.getUCharPatch(x, y, int(dx/ds), int(dy/ds), self.pyr_level)
        return rgb_patch#.transpose(1, 0, 2) # (x, y, c) instead of (y, x ,c)
    # get tumor mask for specified patch
    def getMask(self, x, y, dx, dy) :
        ds = self.ds_mask
        mask = self.mask.getUCharPatch(x, y, int(dx/ds), int(dy/ds), self.pyr_level)
        return mask.reshape(mask.shape[0:2])#.transpose(1, 0, 2)
    # get tissue mask for specified patch
    def getTissue(self, x, y, dx, dy) :
        ds = self.ds_tissue
        tissue = self.tissue.getUCharPatch(x, y, int(dx/ds), int(dy/ds), self.pyr_level)
        return tissue.reshape(tissue.shape[0:2])#.transpose(1, 0, 2)
    # read XML ROI and write TIFF mask
    def XML2TIFF(self, xml_filename, tif_filename, c17 = False):
        camelyon17_type_mask = c17
        self.xml_repository.setSource(xml_filename)
        self.xml_repository.load()
        annotation_mask = mrii.AnnotationToMask()
        label_map = ({'metastases': 1, 'normal': 2}
                     if camelyon17_type_mask
                     else {'_0': 4, '_1': 2, '_2': 1}
                     )
        conversion_order = (['metastases', 'normal']
                            if camelyon17_type_mask
                            else  ['_0', '_1', '_2']
                            )
        annotation_mask.convert(self.annotation_list, tif_filename,
                                self.pyram.getDimensions(),
                                self.pyram.getSpacing(), label_map,
                                conversion_order)
    def XML2TIFF_2(self, xml_filename, maskdir, basename):
        self.xml_repository.setSource(xml_filename)
        self.xml_repository.load()
        conversion_order = ['_0', '_1', '_2']
        annotation_mask = mrii.AnnotationToMask()
        ids = {'_2': 'normal', '_1': 'stressed', '_0': 'tumor'}
        # generate masks
        for id in ids:
            label_map = {id: 1}
            tif_filename = os.path.join(maskdir, ids[id], basename + '_mask.tif')
            annotation_mask.convert(self.annotation_list, tif_filename,
                                    self.pyram.getDimensions(),
                                    self.pyram.getSpacing(), label_map,
                                    conversion_order)
    def XML2TIFF_3(self, xml_filename, maskdir, basename, lab = '_0',
                   suf = 'mask', ids = {'_2': 'normal', '_1':
                                        'stressed', '_0': 'tumor'} ):
        self.xml_repository.setSource(xml_filename)
        self.xml_repository.load()
        conversion_order = ['_0', '_1', '_2']
        annotation_mask = mrii.AnnotationToMask()
        # generate masks
        for id in [lab]:
            label_map = {id: 1}
            tif_filename = os.path.join(maskdir, ids[id], basename + f'_{suf}.tif')
            annotation_mask.convert(self.annotation_list, tif_filename,
                                    self.pyram.getDimensions(),
                                    self.pyram.getSpacing(), label_map,
                                    conversion_order)
