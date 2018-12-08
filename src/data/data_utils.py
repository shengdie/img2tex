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
        
def generate_vocab(tokenlized_fms, cut=0, null='<NULL>', start='<START>', end='<END>',unk='<UNK>'):
    id_to_token, counts = np.unique(tokenlized_fms, return_counts=True)
    #return (id_to_token, counts)
    id_to_token = id_to_token[counts>cut]
    id_to_token = np.delete(id_to_token, np.argwhere(id_to_token==null))
    id_to_token = np.concatenate(([null,start,end, unk], id_to_token))
    token_to_id = {id_to_token[i]: i for i in range(len(id_to_token))}
    return id_to_token, token_to_id