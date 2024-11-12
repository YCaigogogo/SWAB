from PIL import Image
import requests
import torch
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from model.blip.models.blip_retrieval import blip_retrieval
#from models.blip import blip_feature_extractor


image_size = 224
img_path = "/data/yic/clean_code/LOVM-main/new_plot_2.png"
from PIL import Image

def load_demo_image(img_path,image_size,device):
    raw_image = Image.open(img_path).convert('RGB')   
    transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
    image = transform(raw_image).unsqueeze(0).to(device)   
    return image

'''
image=load_demo_image(img_path,image_size,device)

model_path = '/data/yic/model/LOVM/model_base_retrieval_coco.pth'
    
model = blip_feature_extractor(pretrained=model_path, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

caption = 'a woman sitting on the beach with a dog'

multimodal_feature = model(image, caption, mode='multimodal')[0,0]
image_feature = model(image, caption, mode='image')[0,0]
text_feature = model(image, caption, mode='text')[0,0]

print(image_feature)
print(text_feature)
'''

def blip_base_coco():
    return blip_retrieval(pretrained='model/checkpoint/model_base_retrieval_coco.pth', image_size=image_size, vit='base')

def blip_large_coco():
    return blip_retrieval(pretrained='model/checkpoint/model_large_retrieval_coco.pth', image_size=image_size, vit='large')

def blip_base_flickr():
    return blip_retrieval(pretrained='model/checkpoint/model_base_retrieval_flickr.pth', image_size=image_size, vit='base')

def blip_large_flickr():
    return blip_retrieval(pretrained='model/checkpoint/model_large_retrieval_flickr.pth', image_size=image_size, vit='large')

blip_base_coco()