import os
import numpy as np
import cv2

def read2lines(file):
    with open(file) as f:
        content = f.read().splitlines()
    return content

def resize_img(img, tbpad=(1,1), aspect=16):
    """resize image to a aspect aspect by padding, height of images are original_h.
    Arguments:
        original_h {int} -- original height of images
        tbpad {tupple} -- (top, bottom) padding size.
        aspect {int} -- final aspect
    """
    
    oh, ow = img.shape
    new_w = int((oh+tbpad[0]+tbpad[1]) * aspect)
    #o_asp = ow/oh
    if ow < new_w:
        padl = int((new_w - ow)//2)
        padr = int(new_w - padl -ow)
        #print(pad)
        img = cv2.copyMakeBorder(img, tbpad[0],tbpad[1], padl, padr, cv2.BORDER_CONSTANT, value=255)
        #img = cv2.resize(img, (width, height))
        return img
    else:
        return None

def pad_batch_images(imgs, tbpad):
    max_w = max([img.shape[1] for img in imgs])
    n_imgs = []
    for img in imgs:
        oh, ow =  img.shape
        padl = int((max_w - ow) // 2)
        padr = int(max_w -padl - ow)
        n_img = cv2.copyMakeBorder(img, tbpad[0],tbpad[1], padl, padr, cv2.BORDER_CONSTANT, value=255)
        n_imgs.append(n_img)
        

def pad_batch_formulas(formulas, id_pad, id_end, max_len=None):
    """Pad formulas to the max length with id_pad and adds and id_end token
    at the end of each formula

    Args:
        formulas: (list) of list of ints
        max_length: length maximal of formulas

    Returns:
        array: of shape = (batch_size, max_len) of type np.int32
        array: of shape = (batch_size) of type np.int32

    """
    if max_len is None:
        max_len = max(map(lambda x: len(x), formulas))

    batch_formulas = id_pad * np.ones([len(formulas), max_len+1],
            dtype=np.int32)
    formula_length = np.zeros(len(formulas), dtype=np.int32)
    for idx, formula in enumerate(formulas):
        batch_formulas[idx, :len(formula)] = np.asarray(formula,
                dtype=np.int32)
        batch_formulas[idx, len(formula)]  = id_end
        formula_length[idx] = len(formula) + 1

    return batch_formulas, formula_length
        
def generate_vocab(tokenlized_fms, cut=0, null='<NULL>', start='<START>', end='<END>',unk='<UNK>'):
    id_to_token, counts = np.unique(tokenlized_fms, return_counts=True)
    #return (id_to_token, counts)
    id_to_token = id_to_token[counts>cut]
    id_to_token = np.delete(id_to_token, np.argwhere(id_to_token==null))
    #id_to_token = np.concatenate(([null,start,end, unk], id_to_token))
    id_to_token = np.concatenate(([null,end, unk], id_to_token))
    token_to_id = {id_to_token[i]: i for i in range(len(id_to_token))}
    return id_to_token, token_to_id