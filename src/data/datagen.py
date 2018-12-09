import os
import numpy as np
import cv2
from .data_utils import *

class DataGen(object):
    def __init__(self, imgs, formulas):
        self._imgs = imgs
        self._formulas = formulas
        self._len = len(self._formulas)
    def __iter__(self):
        for i in range(self._len):
            yield self._imgs[i], self._formulas[i]
    def __len__(self):
        return self._len

#class DataGen(object):
#    def __init__(self, imgs, formulas, id_pad, id_end, batch_size=20, max_img_aspect=16, tbpad=(1,1), max_formula_len=100):
#        self._imgs = imgs
#        self._formulas = formulas
#        self._batch_size = batch_size
#        self._max_img_aspect = max_img_aspect
#        self._tbpad = tbpad
#        self._max_formula_len = max_formula_len
#        self._id_pad = id_pad
#        self._id_end = id_end
#        self._len = len(self._imgs)
#    def __iter__(self):
#        imgs = []
#        fms = []
#        for i in range(self._len):
#            oh, ow = self._imgs[i]
#            new_w = int((oh+self._tbpad[0]+self._tbpad[1]) * self._max_img_aspect)
#            if ow < new_w and len(self._formulas[i])  < self._max_formula_len:
#                padl = int((new_w - ow)//2)
#                padr = int(new_w - padl -ow)
#                img = cv2.copyMakeBorder(img, self._tbpad[0], self._tbpad[1], padl, padr, cv2.BORDER_CONSTANT, value=255)
#                imgs.append(img)
#                fms.append(self._formulas[i])
#            if len(fms) == self._batch_size:
#                formulas, len_formulas = pad_batch_formulas(fms, self._id_pad, self._id_end)
#                yield imgs, formulas
#                imgs = []
#                fms = []
#        if len(fms) > 0:
#            formulas, len_formulas = pad_batch_formulas(fms, self._id_pad, self._id_end)
#            yield imgs, formulas
#    def __len__(self):
#        return self._len

class Vocab(object):
    def __init__(self, tok_to_id, id_to_tok):
        #self.config = config
        self.tok_to_id = tok_to_id
        self.id_to_tok = id_to_tok
        self.load_vocab()

    def load_vocab(self):
        #special_tokens = [self.config.unk, self.config.pad, self.config.end]
        special_tokens = ['<UNK>', '<NULL>', '<END>']
        #self.tok_to_id = load_tok_to_id(self.config.path_vocab, special_tokens)
        #self.id_to_tok = {idx: tok for tok, idx in self.tok_to_id.items()}
        self.n_tok = len(self.tok_to_id)

        self.id_pad = self.tok_to_id[special_tokens[1]]
        self.id_end = self.tok_to_id[special_tokens[-1]]
        self.id_unk = self.tok_to_id[special_tokens[0]]

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
        #self._start = '<START>'
        self._end = '<END>'
        self._null = '<NULL>'
        self._unk = '<UNK>'
        
        self.img_path = os.path.join(self.data_path, self.image_folder)
        self.formula_path = os.path.join(self.data_path, self.formula_lst)
        self.one2one_path = os.path.join(self.data_path, self.one2one)
        self.images, self.onehot_formulas, self.id_to_token, self.token_to_id = self._get_data()
        self._len = self.images.shape[0]
        
        
        
    def __call__(self, train_portion=0.8, val_portion=0.1):
        train_num = int(train_portion * self._len)
        val_num= int(val_portion * self._len)
        #test_num = self._len - train_num - val_num
        val_end = train_num+val_num

        train_data = DataGen(self.images[:train_num], self.onehot_formulas[:train_num])
        val_data = DataGen(self.images[train_num:val_end], self.onehot_formulas[train_num:val_end])
        test_data = DataGen(self.images[val_end:], self.onehot_formulas[val_end:])
        vocab = Vocab(self.token_to_id, self.id_to_token)
        return train_data, val_data, test_data, vocab
        #return self.images, self.onehot_formulas, self.id_to_token, self.token_to_id

    def __len__(self):
        return self._len

    def _get_data(self):
        formula_idxs, img_names=np.loadtxt(self.one2one_path, dtype=object, unpack=True)

        images = [cv2.imread(os.path.join(self.img_path, f), cv2.IMREAD_GRAYSCALE) for f in img_names]
        #_resizeimg = lambda x: resize_img(x, self.topbotpad, self.aspect)
        #images = list(map(_resizeimg, images))
        max_img_len = (self.topbotpad[0]+self.topbotpad[1]+images[0].shape[0]) * self.aspect
        img_len = [img.shape[1] for img in images]
        argso = np.argsort(img_len)
        #images = [images[i] if images[i].shape[1] < max_img_len else None for i argso]
        formulas = read2lines(self.formula_path)

        #formulas = np.genfromtxt(tex_path, dtype=np.str)
        tokenlized_fm = []
        images_f = []
        for i in argso:
            #if images[i] is not None:
            f = formulas[i].split()
            if len(f) <= self.max_formula_len and images[i].shape[1] <= max_img_len:
            #    f.extend([self._null]*(self.max_formula_len - len(f)))
                tokenlized_fm.append(f)
                images_f.append(images[i])
            #else:
            #    images[i] = None

        #formulas = [formulas[i].split() for i in range(len(images)) if images[i] is not None]
        #images = [e for e in images if e is not None]
        
        #images = np.asarray(images, dtype=np.uint8)

        tokenlized_fm = np.asarray(tokenlized_fm, dtype=np.str)
        #return tokenlized_fm
        id_to_token, token_to_id = generate_vocab(tokenlized_fm, cut=self.min_token_num, 
                                                  null=self._null, #start=self._start, 
                                                  end=self._end,unk=self._unk)
        
        onehot_fm = np.array([[token_to_id[fmt] for fmt in fm] for fm in tokenlized_fm])
        #num_img = onehot_fm.shape[0]
        #onehot_fm = np.concatenate((#np.repeat(token_to_id[self._start], num_img)[:,None], 
        #                            onehot_fm, 
        #                            np.repeat(token_to_id[self._end], num_img)[:,None]), axis=1)
        # reshape to [N, H,W, 1], before [N, H, W]
        #images = np.expand_dims(images, -1)
        #train_images, train_formulas = images[:train_num], onehot_fm[:train_num]
        return images, onehot_fm, id_to_token, token_to_id

#class DataGen(object):
