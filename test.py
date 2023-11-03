import argparse
import os
from PIL import Image
from IPython.display import display
import torch
import torchvision.transforms as transforms 
from torchvision.utils import save_image
from google.colab.patches import cv2_imshow
import cv2
import numpy as np

import AdaIN_net as net



image_size = 512
device = 'cuda'

parser = argparse.ArgumentParser()
parser.add_argument('-content_image', type=str, help='test image')
parser.add_argument('-style_image', type=str, help='style image')
parser.add_argument('-encoder', type=str,)
parser.add_argument('-decoder', type=str, )
parser.add_argument('-alpha', type=float, default=1.0, help='Level of style transfer, value between 0 and 1')
parser.add_argument('-cuda', type=str, help='[y/N]')

opt = parser.parse_args()
content_image = cv2.imread(opt.content_image)

#display(content_image)
#cv2_imshow(content_image)
# Convert the image to a NumPy array with 'uint8' data type
content_image_np = np.array(content_image).astype('uint8')
content_image_size=content_image.shape


# Display the image using cv2_imshow
content_image=Image.open(opt.content_image)

# content_image_size[0],content_image_size[1]=content_image_size[1],content_image_size[0]
#cv2_imshow(content_image_np)
style_image = Image.open(opt.style_image)
output_format = opt.content_image[opt.content_image.find('.'):]
decoder_file = opt.decoder
encoder_file = opt.encoder
alpha = opt.alpha
use_cuda = False
if opt.cuda == 'y' or opt.cuda == 'Y':
	use_cuda = True
out_dir = './output/'
os.makedirs(out_dir, exist_ok=True)

encoder = net.encoder_decoder.encoder
encoder.load_state_dict(torch.load(encoder_file, map_location='cuda'))
decoder = net.encoder_decoder.decoder
decoder.load_state_dict(torch.load(decoder_file, map_location='cuda'))
model = net.AdaIN_net(encoder, decoder)

model.to(device=device)
model.eval()

print('model loaded OK!')
content_image = transforms.Resize(size=(512, 512))(content_image)
style_image = transforms.Resize(size=(512, 512))(style_image)

input_tensor = transforms.ToTensor()(content_image).unsqueeze(0)
style_tensor = transforms.ToTensor()(style_image).unsqueeze(0)




if torch.cuda.is_available() and use_cuda:
	print('using cuda ...')
	model.cuda()
	input_tensor = input_tensor.cuda()
	style_tensor = style_tensor.cuda()
else:
	print('using cpu ...')

out_tensor = None

with torch.no_grad():
	out_tensor = model(input_tensor, style_tensor,opt.alpha)
  

save_file = out_dir + opt.content_image[opt.content_image.rfind('/')+1: opt.content_image.find('.')] \
						+"_style_"+ opt.style_image[opt.style_image.rfind('/')+1: opt.style_image.find('.')] \
						+ "_alpha_" + str(alpha) \
						+ output_format
print('saving output file: ', save_file)
out_tensor=transforms.Resize(size=content_image_size[:2],antialias=True)(out_tensor)
save_image(out_tensor, save_file)
