import numpy as np
from skimage.color import rgb2gray, rgb2hsv, hsv2rgb
from skimage.exposure import adjust_gamma
from skimage.transform import resize

def rng(n=1):
    return np.random.uniform(0,1,n)

def chk_dim_and_chn(im, d=3,c=3):
    if len(im.shape) != d:
        if len(im.shape) == 2:
            #print(f'expected a {d}-dimensional image, received only {len(im.shape)}-dimensional data\ncorrecting by repeating along a third axis')
            w,h = im.shape
            im = im.reshape((w,h,1))
            im = np.repeat(im, 3, 2)
        else:
            raise ValueError(f'expected a {d}-dimensional image, received only {len(im.shape)}-dimensional data')
    elif im.shape[2] != c: raise ValueError(f'expected a {c}-channel image, received only {im.shape[2]} channels.')
    return im

def channel_norm(im, c=3):
    for i in range(c):
        channel = im[:,:,i]
        channel_max = np.max(channel)
        channel_min = np.max(channel)
        if channel_max > channel_min:
            im[:,:,i] = (channel - channel_min) / (channel_max - channel_min)
    return im

def ColorJitter(im, hm=0.8,sm=0.8,vm=0.8,cm=0.8):
    im = chk_dim_and_chn(im)
    # calculate "jitter" values
    h, s, v, c = rng(4) * [hm,sm,vm,cm]
    #print('pre jitter rgb avg:', np.mean(im, axis=(0,1)))
    # convert to hue-sat-val space
    im = rgb2hsv(im)
    # apply jitter
    im += [h,s,v]
    # adjust contrast
    im = adjust_gamma(im, 1+c)
    # convert back to rgb and normalize each channel
    im = hsv2rgb(im)
    im = channel_norm(im)
    #print('post jitter rgb avg:', 255*np.mean(im, axis=(0,1)))
    return hsv2rgb(im)

def Decolorize(im, p=0.5):
    im = chk_dim_and_chn(im)
    if rng() <= p:
        # remember shape
        w,h,c = im.shape
        # calculate luminance matr
        lum = rgb2gray(im)
        # restore 3dim shape
        im = np.repeat(lum.reshape(w,h,1), c, 2)
    return im

def CropResize(im, cl=0.08,ch=1.0,al=3/4,ah=4/3):
    im = chk_dim_and_chn(im)
    # remember original shape
    w,h,c = im.shape
    # target aspect ratio w / h
    aspect = al + (ah-al)*rng()
    target_w = h * aspect
    target_h = w * (1/aspect)
    # crop rectangle and clamp to image size
    crop = cl + (ch-cl)*rng()
    crop_w, crop_h = target_w * crop, target_h * crop
    crop_w, crop_h = min(crop_w, w-1), min(crop_h, h-1)
    # rectangle coordinates
    x1, y1 = (w - crop_w) * rng(), (h - crop_h) * rng()
    x1, y1 = int(x1), int(y1)
    x2, y2 = x1 + crop_w, y1 + crop_h
    x2, y2 = int(x2), int(y2)
    # extract and resize
    im = im[x1:x2,y1:y2,:]
    im = resize(im, (w,h,c))
    return im

def HorizontalFlip(im, p=0.5):
    im = chk_dim_and_chn(im)
    if rng() <= p:
        # flip around second axis
        im = im[:,::-1,:]
    return im

all_fns = [HorizontalFlip,CropResize,Decolorize,ColorJitter]

def apply_all(im, chain=all_fns):
    for A in chain:
        im = A(im)
    return im

no_aug = lambda im: im
