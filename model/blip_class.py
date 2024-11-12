import torch
from model.blip.blip_model import blip_base_coco,blip_base_flickr,blip_large_coco,blip_large_flickr
from transformers import XLMRobertaTokenizer
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.datasets.folder import default_loader
import torch.nn.functional as F

class BLIP():
    def __init__(self, model_size='base', model_ptm_dataset='coco') -> None:
        self.model, self.img_size = self.get_model(model_ptm_dataset, model_size)
        self.model.eval()
        self.model.cuda()

        self.transform = self.get_transform(self.img_size)

        self.loader = default_loader

    def encode_image(self, image):
        with torch.no_grad():
            image_embeds = self.model.visual_encoder(image) 
            image_feat = F.normalize(self.model.vision_proj(image_embeds[:,0,:]),dim=-1)    
            
        return image_feat
    
    def tokenize(self, text, max_len=None):
        raise NotImplementedError
    
    def encode_text(self, caption):
        with torch.no_grad():
            text = self.model.tokenizer(caption, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to('cuda') 
        
            text_output = self.model.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                        return_dict = True, mode = 'text')            
            text_feat = F.normalize(self.model.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1) 

        return text_feat
    
    def get_text_classifier(self, class_names):
        text_feat_list = []
        for class_name in class_names:
            text_feat_list.append(self.encode_text(class_name).squeeze().unsqueeze(0))

        text_classifier = torch.cat(text_feat_list, dim=0)

        return text_classifier


    def get_model(self, dataset='coco', model_size='base'):
        if dataset == 'coco':
            if model_size == 'base':
                model = blip_base_coco()
            elif model_size == 'large':
                model = blip_large_coco()
            img_size = 224
        else:
            if model_size == 'base':
                model = blip_base_flickr()
            elif model_size == 'large':
                model = blip_large_flickr()
            img_size = 224
        
        return model, img_size
    
    
    def get_transform(self, image_size):
        from torchvision.transforms.functional import InterpolationMode
        transform = transforms.Compose([
        transforms.Resize((image_size,image_size),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ]) 
        return transform
    
