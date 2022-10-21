import torch
import copy
from monai.transforms import Randomizable, MapTransform, CropForegroundd
from monai.config import KeysCollection
import numpy as np

from skimage.transform import resize
from torch.nn import ConstantPad2d
import math

from monai.transforms.utils import (
    generate_spatial_bounding_box,
    is_positive,
)
from typing import Callable, Optional, Sequence, Union
from monai.config import IndexSelection
from monai.utils import (
    NumpyPadMode,
    fall_back_tuple
)

from monai.config.type_definitions import NdarrayOrTensor

NumpyPadModeSequence = Union[Sequence[Union[NumpyPadMode, str]], NumpyPadMode, str]

class DeleteNotUsableClinicalData(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
    def __call__(self, data):
        for k in self.keys:
            del data[k]
        return data

class PrepareClinicalData(MapTransform):
    '''
    clinical_keys = ['Age', 'Gender', 'Smoking.History', 'Lobe', 'Diameter.Max.Lung','Diameter.Max.Mediastinum', 'TDR', 'Rebiopsy.diagnosis', 'Average.density..ROI.', 'Margins', 'Pattern.ground.glass', 'Pleural.effusions.biopsy', 'PD.L1', 'TDia', 'NDia', 'MDia', 'Tbio',  'Nbio', 'Mbio', 'Stadium.biopsy', 'PD', 'Recist.ultimo.contatto', 'Dead.Alive', ])
    '''
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        
    def __call__(self, data):
        for key in self.keys:
            del data[key]
        return data

class MyCropForegroundd(CropForegroundd):
    
    def __init__(
        self,
        keys: KeysCollection,
        source_key: str,
        select_fn: Callable = is_positive,
        channel_indices: Optional[IndexSelection] = None,
        margin: Union[Sequence[int], int] = 0,
        k_divisible: Union[Sequence[int], int] = 1,
        mode: NumpyPadModeSequence = NumpyPadMode.CONSTANT,
        start_coord_key: str = "foreground_start_coord",
        end_coord_key: str = "foreground_end_coord",
        allow_missing_keys: bool = False,
        **np_kwargs,
    ) -> None:
        super().__init__(keys, source_key, select_fn, channel_indices, margin, k_divisible, mode, start_coord_key, end_coord_key, allow_missing_keys, **np_kwargs,)
        self.cropper.compute_bounding_box = self._compute_bounding_box
        
    def compute_squared_spatial_size(self, spatial_shape: Sequence[int], k: Union[Sequence[int], int]):
        """
        Compute the target spatial size which should be divisible by `k` and should be squared.
    
        Args:
            spatial_shape: original spatial shape.
            k: the target k for each spatial dimension.
                if `k` is negative or 0, the original size is preserved.
                if `k` is an int, the same `k` be applied to all the input spatial dimensions.
    
        """
        k = fall_back_tuple(k, (1,) * len(spatial_shape))
        new_size = []
        for k_d, dim in zip(k, spatial_shape):
            new_dim = int(np.ceil(dim / k_d) * k_d) if k_d > 0 else dim
            new_size.append(new_dim)
        
        new_size[0] = new_size[1] = np.asarray(new_size[:-1]).max()
        
        return new_size
      
    def _compute_bounding_box(self, img: NdarrayOrTensor):
        #Sistemare questo codice per me
        
        box_start, box_end = generate_spatial_bounding_box(img, self.cropper.select_fn, self.cropper.channel_indices, self.cropper.margin)
        box_start_ = np.asarray(box_start, dtype=np.int16)
        box_end_ = np.asarray(box_end, dtype=np.int16)
        orig_spatial_size = box_end_ - box_start_
        # make the spatial size divisible by `k`
        spatial_size = np.asarray(self.compute_squared_spatial_size(spatial_shape=orig_spatial_size, k=self.cropper.k_divisible))
        # update box_start and box_end
        box_start_ = box_start_ - np.floor_divide(np.asarray(spatial_size) - orig_spatial_size, 2)
        box_end_ = box_start_ + spatial_size
        return box_start_, box_end_

class ExpandDims(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
    def __call__(self, data):
        for key in self.keys:
            if len(data[key].shape)<4:
                data[key] = np.expand_dims(data[key], -1)
        return data

class AsDepthFirstD(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
    def __call__(self, data):
        for key in self.keys:
            data[key]= np.moveaxis(data[key], -1, 0)
        return data

class ConvertListToTensor(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
    def __call__(self, data):
        for key in self.keys:
            data[key]=torch.tensor(data[key])
        return data
    
class ConcatChDim(MapTransform):
    def __init__(self, keys: KeysCollection, deleteLastDimIf1 = False):
        super().__init__(keys)
        self.deleteLastDimIf1 =  deleteLastDimIf1
    def __call__(self, data):
        img = np.concatenate((data[self.keys[0]], data[self.keys[1]]), axis = 0)
        if self.deleteLastDimIf1:
            img = np.squeeze(img, axis = -1)
        data['patched_img'] = img
        del data[self.keys[1]]
        del data['image']
        del data['mask']
        del data['att_maps_meta_dict']
        del data['patched_img_meta_dict']
        return data
    
class DeleteKeys(MapTransform):
    def __init__(self, keys: KeysCollection, del_keys = [], deleteLastDimIf1 = False):
        super().__init__(keys)
        self.deleteLastDimIf1 =  deleteLastDimIf1
        self.del_keys = del_keys
    def __call__(self, data):
        img = data[self.keys[0]]
        if self.deleteLastDimIf1:
            img = np.squeeze(img, axis = -1)
        data['patched_img'] = img
        for k in self.del_keys:
            del data[k]
        return data

'''Transformations for all slices patched'''
class ResizeWithRatioDVariableDim(MapTransform):
    def __init__(self, keys: KeysCollection, image_size):
        super().__init__(keys)
        self.image_size = image_size
        
    def __call__(self, data):
        d = copy.deepcopy(data)
        for key in self.keys:
            num_slices = d[key].shape[3]
            num_patches = math.pow(math.ceil(math.sqrt(num_slices)),2)
            num_cols = int(math.sqrt(num_patches))
            var_size = self.image_size//num_cols
            if d[key].shape[1]>d[key].shape[2]:
                h = var_size
                hpercent = (h/float(d[key].shape[1]))
                w = int(float(d[key].shape[2])*float(hpercent))
            elif d[key].shape[1]<d[key].shape[2]:
                w = var_size
                wpercent = (w/float(d[key].shape[2]))
                h = int(float(d[key].shape[1])*float(wpercent))
            else:
                h = var_size
                w = var_size
            d[key] = resize(d[key], (d[key].shape[0],h,w,d[key].shape[-1]), anti_aliasing=True)
            d[key]=d[key].astype(float)
        return d
    
class TensorPad(MapTransform):
    def __init__(self, keys: KeysCollection, image_size, pad_value = 0):
        super().__init__(keys)
        self.image_size = image_size
        self.pad_value = pad_value
        
    def __call__(self, data):
        for key in self.keys:
            pad_left = 0
            pad_right = 0
            h_diff = self.image_size - data[key].shape[1]
            w_diff = self.image_size - data[key].shape[2]
            pad_top = h_diff // 2 if h_diff > 0 else 0
            pad_bot = h_diff // 2 if h_diff > 0 else 0
            if h_diff % 2 != 0:
                pad_top = pad_top + 1
                
            pad_left = w_diff // 2 if w_diff > 0 else 0
            pad_right = w_diff // 2 if w_diff > 0 else 0
            if w_diff % 2 != 0:
                pad_left = pad_left + 1
            padding_func = ConstantPad2d((pad_left, pad_right, pad_top, pad_bot),self.pad_value)
            data[key] = padding_func(data[key])
        return data
    
class PatchedImageAllSlice(MapTransform):
    def __init__(self, keys: KeysCollection, img_size = 512):
        super().__init__(keys)
        self.start_id = 0
    
    def __call__(self, data):
        img_key, label_key = self.keys
        num_slices = data[img_key].shape[3]
        num_patches = math.pow(math.ceil(math.sqrt(num_slices)),2)
        num_cols = int(math.sqrt(num_patches))
        patch_row = data[img_key].shape[1]
        patch_col = data[img_key].shape[2]
        patched_img = torch.zeros(data[img_key].shape[0], patch_row*num_cols,patch_col*num_cols)
                
        counter = 0
        for i in range(num_cols):
            for j in range(num_cols):
                if counter < num_slices:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,:,:, self.start_id+counter]
                    counter += 1
                else:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,:,:,self.start_id+counter-1]
       
        cropper = copy.deepcopy(data)
        cropper[img_key] = patched_img
        del cropper[label_key]
        return cropper
    
''''''

class ResizeWithRatioD(MapTransform):
    def __init__(self, keys: KeysCollection, image_size):
        super().__init__(keys)
        self.image_size = image_size
        
    def __call__(self, data):
        d = copy.deepcopy(data)
        for key in self.keys:
            if d[key].shape[1]>d[key].shape[2]:
                h = self.image_size
                hpercent = (h/float(d[key].shape[1]))
                w = int(float(d[key].shape[2])*float(hpercent))
            elif d[key].shape[1]<d[key].shape[2]:
                w = self.image_size
                wpercent = (w/float(d[key].shape[2]))
                h = int(float(d[key].shape[1])*float(wpercent))
            else:
                h = self.image_size
                w = self.image_size
            d[key] = resize(d[key], (d[key].shape[0],h,w,d[key].shape[-1]), anti_aliasing=True)
            d[key]=d[key].astype(float)
        return d

class PatchedImage(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        self.start_id = 0
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
    
    def __call__(self, data):
        img_key, label_key = self.keys
        num_slices = data[img_key].size(3)
        patch_row = data[img_key].size()[1]
        patch_col = data[img_key].size()[2]
        patched_img = torch.zeros(data[img_key].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        if num_slices <= self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
                
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,:,:, self.start_id+counter]
                    counter += 1
                else:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,:,:,self.start_id+counter-1]
       
        cropper = copy.deepcopy(data)
        cropper[img_key] = patched_img
        cropper['start_id'] = self.start_id
        del cropper[label_key]
        return cropper
    
class CenterPatchedImage(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        self.start_id = 0
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    
    def __call__(self, data):
        img_key, label_key = self.keys
        num_slices = data[img_key].size(3)
        patch_row = data[img_key].size()[1]
        patch_col = data[img_key].size()[2]
        patched_img = torch.zeros(data[img_key].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        if num_slices <= self.num_patches:
            self.start_id = 0
        else:
            self.start_id = int(num_slices/2)-int(self.num_patches/2)
                
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,:,:, self.start_id+counter]
                    counter += 1
                else:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,:,:,self.start_id+counter-1]
       
        cropper = copy.deepcopy(data)
        cropper[img_key] = patched_img
        cropper['start_id'] = self.start_id
        cropper['num_slices'] = num_slices
        del cropper[label_key]
        return cropper
    
    
class ListPatchedImage(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, num_patches=9, stride = 2):
        super().__init__(keys)
        self.num_patches = num_patches
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        self.stride = stride
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def __call__(self, data):
        img_key, label_key = self.keys
        num_slices = data[img_key].size(3)
        patch_row = data[img_key].size()[1]
        patch_col = data[img_key].size()[2]
        num_input = math.ceil((num_slices-self.num_patches)/self.stride)
        if not num_input > 0:
            num_input = 1
        if num_input % 2 == 0:
            num_input += 1
        list_patched_imgs = []
        start_id = 0
        for n in range(num_input):
            patched_img = torch.zeros(data[img_key].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
            counter = 0
            for i in range(self.num_cols):
                for j in range(self.num_cols):
                    if counter < num_slices-start_id:
                        patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,:,:, start_id+counter]
                        counter += 1
                    else:
                        patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,:,:,start_id+counter-1]
            
            list_patched_imgs.append(patched_img)
            start_id = start_id+self.stride
        
        cropper = copy.deepcopy(data)
        cropper[img_key] = torch.stack(list_patched_imgs)
        cropper['num_slices'] = num_slices
        del cropper[label_key]
        return cropper
    
class Transpose_BWImage(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            data[key]=data[key].transpose()
        return data

class TransposeITKD(MapTransform):
    def __call__(self, data):
        for key in self.keys:
            data[key]=data[key].transpose((1,2,0))
        return data
    
class CorrectSpacing(MapTransform):
    def __call__(self, data):
        i,m = self.keys
        if not np.all(data[f'{m}_meta_dict']['spacing'] == data[f'{i}_meta_dict']['spacing']):
            data[f'{i}_meta_dict']['spacing'] = data[f'{m}_meta_dict']['spacing']
            data[f'{i}_meta_dict']['original_affine'] = data[f'{m}_meta_dict']['original_affine']
            data[f'{i}_meta_dict']['affine'] = data[f'{m}_meta_dict']['affine']
        if 'ITK_non_uniform_sampling_deviation' in data[f'{i}_meta_dict']:
            data[f'{i}_meta_dict'].pop('ITK_non_uniform_sampling_deviation')  
        return data    

class RandPatchedImageWith0Padding(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key, label_key = self.keys
        patch_row = data[img_key].size()[2]
        patch_col = data[img_key].size()[3]
        patched_img = torch.zeros(data[img_key].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,self.start_id+counter,:,:]
                    counter += 1
       
        cropper = copy.deepcopy(data)
        cropper[img_key] = patched_img
        cropper['start_id'] = self.start_id
        return cropper
    
class RandPatchedImage(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key, label_key = self.keys
        patch_row = data[img_key].size()[2]
        patch_col = data[img_key].size()[3]
        patched_img = torch.zeros(data[img_key].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key][:,self.start_id+counter-1,:,:]
       
        cropper = copy.deepcopy(data)
        cropper[img_key] = patched_img
        cropper['start_id'] = self.start_id
        return cropper

class RandPatchedImageLateFusion(Randomizable, MapTransform):
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
       
        cropper = copy.deepcopy(data)
        cropper[img_key_T1] = patched_img_T1
        cropper[img_key_T2] = patched_img_T2
        cropper['start_id'] = self.start_id
        return cropper

class RandPatchedImageAndEarlyFusion(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
       
        patched_fused_img = torch.cat((patched_img_T1, patched_img_T2), dim = 0)
        cropper = copy.deepcopy(data)
        del cropper[img_key_T1]
        del cropper[img_key_T2]
        cropper['fusedImage'] = patched_fused_img
        cropper['start_id'] = self.start_id
        return cropper

class CenterPatchedImageAndEarlyFusion(MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
      
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else:
            self.start_id = int(num_slices/2)-int(self.num_patches/2)
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
       
        patched_fused_img = torch.cat((patched_img_T1, patched_img_T2), dim = 0)
        cropper = copy.deepcopy(data)
        del cropper[img_key_T1]
        del cropper[img_key_T2]
        cropper['fusedImage'] = patched_fused_img
        cropper['start_id'] = self.start_id
        return cropper
        

class RandDepthCrop(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_slices=9):
        super().__init__(keys)
        self.num_slices = num_slices
        self.start_id = 0
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key, label_key = self.keys
        max_value = data[img_key].shape[1] - self.num_slices
        self.randomize(max_value)
        slice_ = data[img_key][0,:,:,self.start_id:(self.start_id+self.num_slices)]
        n = slice_.shape[0]
        while n<self.num_slices:
            slice_ = torch.cat([slice_, slice_[-1].unsqueeze(0)],dim = 0)
            n+=1
        cropper = copy.deepcopy(data)
        cropper[img_key] = slice_
        cropper['start_id'] = self.start_id
        return cropper
    
class NewMergedImage(MapTransform):
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
    
    def __call__(self, data):
        img_key_T1, img_key_T2 = self.keys
        img_key_merge = 'merged'
        
        merged_data = data[img_key_T1] - data[img_key_T2]
        data[img_key_merge] = merged_data
        return data
    
class RandPatchedImage3Channels(Randomizable, MapTransform):
    
    def __init__(self, keys: KeysCollection, num_patches=9):
        super().__init__(keys)
        self.num_patches = num_patches
        self.start_id = 0
        self.num_cols = int(math.sqrt(self.num_patches))
        self.dim = 1
        assert (self.num_cols**2 == self.num_patches), 'Error in num_slices values'
    
    def randomize(self, max_value):
        self.start_id = int(self.R.random_sample() * max_value)
        
    def __call__(self, data):
        img_key_T1, img_key_T2, label_key = self.keys
        img_key_merge = 'merged'
        patch_row = data[img_key_T1].size()[2]
        patch_col = data[img_key_T1].size()[3]
        patched_img_T1 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_img_T2 = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        patched_merge = torch.zeros(data[img_key_T1].size()[0], patch_row*self.num_cols,patch_col*self.num_cols)
        num_slices = data[img_key_T1].size()[1]
        
        if num_slices < self.num_patches:
            self.start_id = 0
        else: self.randomize(num_slices - self.num_patches -1 )
        
        counter = 0
        for i in range(self.num_cols):
            for j in range(self.num_cols):
                if counter < num_slices:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter,:,:]
                    patched_merge[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_merge][:,self.start_id+counter,:,:]
                    counter += 1
                else:
                    patched_img_T1[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T1][:,self.start_id+counter-1,:,:]
                    patched_img_T2[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_T2][:,self.start_id+counter-1,:,:]
                    patched_merge[0, i*patch_row:(i*patch_row+patch_row), j*patch_col:(j*patch_col+patch_col)] = data[img_key_merge][:,self.start_id+counter-1,:,:]
                    
        patched_fused_img = torch.cat((patched_img_T1, patched_img_T2, patched_merge), dim = 0)
        cropper = copy.deepcopy(data)
        del cropper[img_key_T1]
        del cropper[img_key_T2]
        del cropper[img_key_merge]
        cropper['fusedImage'] = patched_fused_img
        cropper['start_id'] = self.start_id
        return cropper
    
class NDITKtoNumpy(MapTransform):
    def __init__(self, keys: KeysCollection):
        super().__init__(keys)
        
    def __call__(self, data):
        for k in self.keys:
            data[k] = np.asarray(data[k])
        return data