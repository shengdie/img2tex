import os
import numpy as np
import cv2
from .data_utils import *

class LoadData(object):
    
    def __init__(self, data_path, image_folder, formula_lst, one2one, min_token_num=0, max_formula_len=100, 
                topbotpad=(1,1), aspect=16):
        self.data_path = data_path
        self.image_folder = image_folder
        self.formula_lst = formula_lst
        self.one2one = one2one
        self.min_token_num = min_token_num
        self.max_formula_len = max_formula_len
        self.topbotpad = topbotpad
        self.aspect = aspect
        self._start = '<START>'
        self._end = '<END>'
        self._null = '<NULL>'
        self._unk = '<UNK>'
        
        self.img_path = os.path.join(self.data_path, self.image_folder)
        self.formula_path = os.path.join(self.data_path, self.formula_lst)
        self.one2one_path = os.path.join(self.data_path, self.one2one)
        self.images, self.onehot_formulas, self.id_to_token, self.token_to_id = self._get_data()
        
    def __call__(self):
        return self.images, self.onehot_formulas, self.id_to_token, self.token_to_id

    def _get_data(self):
        #img_path = os.path.join(self., image_folder)
        #tex_path = os.path.join(base_path, tex_file)
        #one2one_path = os.path.join(base_path, one2one)
        formula_idxs, img_names=np.loadtxt(self.one2one_path, dtype=object, unpack=True)
        images = []
        images = [cv2.imread(os.path.join(self.img_path, f), cv2.IMREAD_GRAYSCALE) for f in img_names]
        _resizeimg = lambda x: resize_img(x, self.topbotpad, self.aspect)
        images = list(map(_resizeimg, images))
        #for i in range(len(images)):

        formulas = read2lines(self.formula_path)

        #formulas = np.genfromtxt(tex_path, dtype=np.str)
        tokenlized_fm = []
        for i in range(len(images)):
            if images[i] is not None:
                f = formulas[i].split()
                if len(f) < self.max_formula_len:
                    f.extend([self._null]*(self.max_formula_len - len(f)))
                    tokenlized_fm.append(f)
                else:
                    images[i] = None

        #formulas = [formulas[i].split() for i in range(len(images)) if images[i] is not None]
        images = [e for e in images if e is not None]
        #return images
        images = np.asarray(images, dtype=np.uint8)

        tokenlized_fm = np.asarray(tokenlized_fm, dtype=np.str)
        #return tokenlized_fm
        id_to_token, token_to_id = generate_vocab(tokenlized_fm, cut=self.min_token_num, 
                                                  null=self._null, start=self._start, end=self._end,unk=self._unk)
        #return id_to_token, token_to_id
        onehot_fm = np.array([[token_to_id[fmt] for fmt in fm] for fm in tokenlized_fm])
        num_img = onehot_fm.shape[0]
        onehot_fm = np.concatenate((np.repeat(token_to_id[self._start], num_img)[:,None], onehot_fm, 
                                    np.repeat(token_to_id[self._end], num_img)[:,None]), axis=1)
        num_total = images.shape[0]
        train_num = int(0.8 * num_total)
        val_num= int(0.1 * num_total)
        test_num = num_total - train_num - val_num
        #train_images, train_formulas = images[:train_num], onehot_fm[:train_num]
        return images, onehot_fm, id_to_token, token_to_id