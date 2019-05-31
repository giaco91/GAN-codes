import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image,ExifTags,ImageFilter,ImageOps
import PIL
import numpy as np

import matplotlib.pyplot as plt

def save_comparison_plot(img_corrupted,img_inpainted,img_orig,save_path):
    fig = plt.figure()
    ax=plt.subplot(131)
    ax.set_axis_off()
    ax.set_title('corrupted')
    plt.imshow(img_corrupted)
    ax=plt.subplot(132)
    ax.set_axis_off()
    ax.set_title('inpainted')
    plt.imshow(img_inpainted)
    ax=plt.subplot(133)
    ax.set_axis_off()
    ax.set_title('original')
    plt.imshow(img_orig)
    plt.subplots_adjust(hspace=-1, wspace=0)
    plt.savefig(save_path+'.png', format='png', dpi='figure')   
    plt.close('all')

def img_show(im):
    im.show()

def augment(img):
    return

def get_mask(imsize,N=5,S=4):
    img=create_image(imsize,imsize)
    rand_coord=np.random.randint(imsize-S, size=(N, 2))
    img_px=img.load()
    for n in range(N):
        for sx in range(S):
            for sy in range(S):
                # print(img_px[int(rand_coord[n,0]),int(rand_coord[n,1])])
                img_px[int(rand_coord[n,0])+sx,int(rand_coord[n,1])+sy]=(0,0,0)
    return img

# Create a new image with the given size
def create_image(i, j):
  image = Image.new("RGB", (i, j), "white")
  return image

def img_make_mask(im):
    #sets all rgb values that are non-zero to one
    im_rgb=im.convert('RGB')
    W,H=im_rgb.size
    new_img=create_image(W,H)
    im_rgb_px=im_rgb.load()
    # for w in range(0,W):
    #     for h in range(0,H):
    #         if im_rgb_px[w,h][0]+im_rgb_px[w,h][1]+im_rgb_px[w,h][2]>10:
    #             im_rgb_px[w,h]=(256,256,256)
    new_img_px=new_img.load()
    for w in range(W):
        for h in range(H):
            if im_rgb_px[w,h][0]+im_rgb_px[w,h][1]+im_rgb_px[w,h][2]<50:
                new_img_px[w,h]=(0,0,0)
    return new_img

def correct_for_rotation(img):
    if img._getexif() is not None:
        exif=dict((ExifTags.TAGS[k], v) for k, v in img._getexif().items() if k in ExifTags.TAGS)
        if 'Orientation' in exif.keys():
            if not exif['Orientation']:
                return img
            elif exif['Orientation']==3:
                print('rotating image by 180')
                img=img.rotate(180, expand=True)
            elif exif['Orientation']==6:
                print('rotating image by 270')
                img=img.rotate(270, expand=True)
            elif exif['Orientation']==8:    
                print('rotating image by 90')
                img=img.rotate(90, expand=True)
    return img

def get_pixel(image, i, j):
  # Inside image bounds?
  width, height = image.size
  if i > width or j > height:
    return None

  # Get Pixel
  pixel = image.getpixel((i, j))
  return pixel

def square_crop(img_path):
    image = load(img_path).convert("RGB")
    w,h=image.size
    if w<h:
        marg_up=round((h-w)/2)
        marg_down=h-w-marg_up
        return image.crop((0, marg_down, w, h-marg_up))
    else:
        marg_left=round((w-h)/2)
        marg_right=w-h-marg_left
        return image.crop((marg_left,0,w-marg_right,h))

def resize_to_height_ref(image,n_height):
    w,h=image.size
    return image.resize((n_height,round(n_height*h/w)),Image.ANTIALIAS)

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.

    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [torch.from_numpy(x) for x in images_np]
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =1, factor=1, interpolation='lanczos',show=True,save_path=None):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='gray', interpolation=interpolation)
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path+'.png', format='png', dpi='figure')
    plt.close('all')
    
    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 

    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]#adds one dimension for batchsize=1

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter,optimizer=None):
    """Runs optimization loop.

    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)
        return optimizer

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        if optimizer is None:
            optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            optimizer.zero_grad()
            print('optimizer in loop '+str(j))
            closure()
            optimizer.step()
        return optimizer
    else:
        assert False





