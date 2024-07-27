import os
import errno
from PIL import Image
from array import *
from random import shuffle

Image.MAX_IMAGE_PIXELS = None

def resize_image(image, max_size=(28, 28)):
    """Resize an image so that its maximum dimensions do not exceed max_size."""
    return image.resize(max_size, Image.Resampling.LANCZOS)
     
def resize_and_pad_image(image, target_size=(28, 28), background_color=0):
    """
    Resize an image to the target size and pad with background color
    if the original aspect ratio is maintained.
    """
    # First, resize the image while maintaining the aspect ratio.
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    
    # Create a new square image of the target size with a background color.
    new_image = Image.new('L', target_size, background_color)
    
    # Paste the resized image onto the center of the square background.
    image_width, image_height = image.size
    top_left_corner = ((target_size[0] - image_width) // 2, (target_size[1] - image_height) // 2)
    new_image.paste(image, top_left_corner)
    
    return new_image

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

# Load from and save to
mkdir_p('5_Mnist')
# Names = [['4_Png\Train','5_Mnist\\train'],['4_Png\Test','5_Mnist\\t10k']]
Names = [[r'4_Png\Train', r'5_Mnist\train'], [r'4_Png\Test', r'5_Mnist\t10k']]


for name in Names:	
    data_image = array('B')
    data_label = array('B')

    FileList = []
    for dirname in os.listdir(name[0]): 
        path = os.path.join(name[0],dirname)
        for filename in os.listdir(path):
            if filename.endswith(".png"):
                FileList.append(os.path.join(name[0],dirname,filename))

    shuffle(FileList) # Usefull for further segmenting the validation set


    for filename in FileList:
        # print (filename)
        label = int(filename.split('\\')[2])
        Im = Image.open(filename)
        # print("Unresized image size:", Im.size) 
        Im=resize_and_pad_image(Im)
        # print("Resized image size:", Im.size) 
        pixel = Im.load()
        width, height = Im.size
        for x in range(0,width):
            for y in range(0,height):
                data_image.append(pixel[x,y])
        data_label.append(label) # labels start (one unsigned byte each)
    hexval = "{0:#0{1}x}".format(len(FileList),6) # number of files in HEX
    hexval = '0x' + hexval[2:].zfill(8)
    
    # header for label array
    header = array('B')
    header.extend([0,0,8,1])
    header.append(int('0x'+hexval[2:][0:2],16))
    header.append(int('0x'+hexval[2:][2:4],16))
    header.append(int('0x'+hexval[2:][4:6],16))
    header.append(int('0x'+hexval[2:][6:8],16))	
    data_label = header + data_label

    # additional header for images array	
    if max([width,height]) <= 256:
        header.extend([0,0,0,width & 0xFF,0,0,0,height & 0xFF])
    else:
        raise ValueError('Image exceeds maximum size: 256x256 pixels');

    header[3] = 3 # Changing MSB for image data (0x00000803)	
    data_image = header + data_image
    output_file = open(name[1]+'-images-idx3-ubyte', 'wb')
    data_image.tofile(output_file)
    output_file.close()
    output_file = open(name[1]+'-labels-idx1-ubyte', 'wb')
    data_label.tofile(output_file)
    output_file.close()

# gzip resulting files
for name in Names:
    os.system('gzip '+name[1]+'-images-idx3-ubyte')
    os.system('gzip '+name[1]+'-labels-idx1-ubyte')