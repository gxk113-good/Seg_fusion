###########################################################################
# Created by: Hang Zhang 
# Email: zhang.hang@rutgers.edu 
# Copyright (c) 2017
###########################################################################

import random
import numpy as np

import torch
import torch.utils.data as data

from PIL import Image, ImageOps, ImageFilter

__all__ = ['BaseDataset', 'test_batchify_fn']

class BaseDataset(data.Dataset):
    def __init__(self, root, split, mode=None, transform=None, 
                 target_transform=None, base_size=520, crop_size=480):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.mode = mode if mode is not None else split
        self.base_size = base_size
        self.crop_size = crop_size
        if self.mode == 'train':
            print('BaseDataset: base_size {}, crop_size {}'. \
                format(base_size, crop_size))

    def __getitem__(self, index):
        raise NotImplemented

    @property
    def num_class(self):
        return self.NUM_CLASS

    @property
    def pred_offset(self):
        raise NotImplemented

    def make_pred(self, x):
        return x + self.pred_offset

    def _val_sync_transform(self, ir_img, vis_img, mask):
        outsize = self.crop_size
        short_size = outsize
        w, h = ir_img.size
        if w > h:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        ir_img = ir_img.resize((ow, oh), Image.BILINEAR)
        vis_img = vis_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = ir_img.size
        x1 = int(round((w - outsize) / 2.))
        y1 = int(round((h - outsize) / 2.))
        ir_img = ir_img.crop((x1, y1, x1+outsize, y1+outsize))
        vis_img = vis_img.crop((x1, y1, x1+outsize, y1+outsize))
        mask = mask.crop((x1, y1, x1+outsize, y1+outsize))
        # final transform
        return ir_img, vis_img, self._mask_transform(mask)

    def _sync_transform(self, ir_img, vis_img, mask):
        # random mirror
        if random.random() < 0.5:
            # img = img.transpose(Image.FLIP_LEFT_RIGHT)
            ir_img = ir_img.transpose(Image.FLIP_LEFT_RIGHT)
            vis_img = vis_img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        crop_size = self.crop_size
        # random scale (short edge from 480 to 720)
        short_size = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
        w, h = ir_img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        # img = img.resize((ow, oh), Image.BILINEAR)
        ir_img = ir_img.resize((ow, oh), Image.BILINEAR)
        vis_img = vis_img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < crop_size:
            padh = crop_size - oh if oh < crop_size else 0
            padw = crop_size - ow if ow < crop_size else 0
            # img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            ir_img = ImageOps.expand(ir_img, border=(0, 0, padw, padh), fill=0)
            vis_img = ImageOps.expand(vis_img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
        # random crop crop_size
        w, h = ir_img.size
        x1 = random.randint(0, w - crop_size)
        y1 = random.randint(0, h - crop_size)
        # img = img.crop((x1, y1, x1+crop_size, y1+crop_size))
        ir_img = ir_img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        vis_img = vis_img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
        mask = mask.crop((x1, y1, x1+crop_size, y1+crop_size))
        # gaussian blur as in PSP
        if random.random() < 0.5:
            # img = img.filter(ImageFilter.GaussianBlur(
            #     radius=random.random()))
            ir_img = ir_img.filter(ImageFilter.GaussianBlur(radius=random.random()))
            vis_img = vis_img.filter(ImageFilter.GaussianBlur(radius=random.random()))
        # final transform
        return ir_img, vis_img, self._mask_transform(mask)            #img, self._mask_transform(mask)

    def _mask_transform(self, mask):
        return torch.from_numpy(np.array(mask)).long()


def test_batchify_fn(data):
    error_msg = "batch must contain tensors, tuples or lists; found {}"
    if isinstance(data[0], (str, torch.Tensor)):
        return list(data)
    elif isinstance(data[0], (tuple, list)):
        data = zip(*data)
        return [test_batchify_fn(i) for i in data]
    raise TypeError((error_msg.format(type(data[0]))))
