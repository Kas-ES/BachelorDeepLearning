import glob
import os

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS = [512, 512, 3]
Mask_Path = 'Dataset_CCE/Frame_Mask_for_NoPolyp_CCE.png'

mask = imread(Mask_Path)
mask = mask[:, :, :3]

Img_Path = 'Dataset_CCE/NoPolyp'
Mask_List = sorted(next(os.walk(Img_Path))[2])


def Img_Data_Ext(mask, Mask_List, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
                 Mask_Path, Img_Path, Img_Extension, Mask_Extension):
    Img = np.zeros((len(Mask_List), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
    Mask = np.zeros((len(Mask_List), IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.uint8)
    n = 0
    TEXT = '{}/*.' + Mask_Extension  # 'png'
    TEXT1 = '{}/{}.' + Img_Extension  # 'png'
    for img_path in glob.glob(TEXT.format(Img_Path)):
        base = os.path.basename(img_path)
        image_ID, ext = os.path.splitext(base)
        image_path = TEXT1.format(Img_Path, image_ID)
        image = imread(image_path)
        image = image[:, :, :3]
        # Temporary image variable
        TempI = resize(image[:, :, :IMG_CHANNELS],
                       (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                       mode='constant',
                       anti_aliasing=True,
                       preserve_range=True)
        ResizedImage = np.multiply(TempI, mask)
        imsave('Dataset_CCE/NoPolypImageResized/' + "NoPolyp" + str(n) + ".png", ResizedImage)
        n += 1

    return Img, Mask

# images, masks = Img_Data_Ext(mask, Mask_List, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
#                               Mask_Path, Img_Path, 'jpg', 'jpg')

## Saving the masks
# b = 0
# for img in masks:
#     imsave('Dataset_CCE/NoPolyMaskResized/'+"NoPolyp"+str(b)+"_mask"+".png", img)
#     b += 1


##Rename and saving the images and masks
# path = "Dataset_CCE/Inputs_Train"
# modifiedPath = "Dataset_CCE/Modified_inputs_Train"
# #Sorted is a awful method that is not guranteed to work, use natsorted instead
# files = natsorted(os.listdir(path))
# files = natsorted(files)
#
# a = 1
# for index, filename in enumerate(files):
#     os.rename(os.path.join(path, filename), os.path.join(modifiedPath, 'tr'+str(index+1)+'.png'))
#     a += 1
