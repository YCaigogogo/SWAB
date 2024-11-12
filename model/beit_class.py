import torch
from model.beit.modeling_finetune import beit3_base_patch16_384_retrieval, beit3_large_patch16_384_retrieval
from transformers import XLMRobertaTokenizer
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.datasets.folder import default_loader

class BEIT3():
    def __init__(self, model_size='base', model_ptm_dataset='coco') -> None:
        self.model, self.img_size = self.get_model(model_size)
        self.ckp_path = self.get_ckp(model_size, model_ptm_dataset)
        model_key ='model|module'
        model_prefix =''
        self.load_model_and_may_interpolate(self.ckp_path, self.model, model_key, model_prefix)
        self.model.eval()
        self.model.cuda()
        self.tokenizer = XLMRobertaTokenizer("/data/yic/model/LOVM/beit3.spm")
        self.num_max_bpe_tokens = 64
        self.transform = self.get_transform(self.img_size)

        self.bos_token_id = self.tokenizer.bos_token_id
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.loader = default_loader

    def encode_image(self, image):
        with torch.no_grad():
            vision_cls, _ = self.model(image=image, only_infer=True)

        return vision_cls
    
    def tokenize(self, text, max_len=None):
        if isinstance(text, str):
            tokens = self.tokenizer(text, return_tensors="pt").input_ids
            tokens = tokens.tolist()[0]
        else:
            tokens = text[:]
        if len(tokens) == 0:
            raise RuntimeError("The text segment should contains at least one tokens!")
        if max_len is None:
            max_len = self.num_max_bpe_tokens

        if len(tokens) > max_len - 2:
            tokens = tokens[:max_len - 2]

        num_tokens = len(tokens)
        padding_mask = [0] * num_tokens + [1] * (max_len - num_tokens)
        return tokens + [self.pad_token_id] * (max_len - num_tokens), padding_mask, num_tokens
    
    def encode_text(self, text):
        with torch.no_grad():
            input_tokens, padding_mask, num_tokens = self.tokenize(text)
            input_tokens = torch.tensor(input_tokens)
            input_tokens = input_tokens.unsqueeze(0)
            input_tokens = input_tokens.cuda()
            padding_mask = torch.tensor(padding_mask)
            padding_mask = padding_mask.unsqueeze(0)
            padding_mask = padding_mask.cuda()
            _, language_cls = self.model(text_description=input_tokens, padding_mask=padding_mask, only_infer=True)

        return language_cls
    
    def get_text_classifier(self, class_names):
        text_feat_list = []
        for class_name in class_names:
            text_feat_list.append(self.encode_text(class_name))

        text_classifier = torch.cat(text_feat_list, dim=0)

        return text_classifier


    def get_model(self, model_size='base'):
        if model_size == 'base':
            model = beit3_base_patch16_384_retrieval()
        elif model_size == 'large':
            model = beit3_large_patch16_384_retrieval()
        img_size = 384
        
        return model, img_size
    
    def get_ckp(self, model_size='base', model_ptm_dataset='coco'):
        if model_size == 'base':
            if model_ptm_dataset == 'coco':
                ckp_path = "model/checkpoint/beit3_base_patch16_384_coco_retrieval.pth"
            elif model_ptm_dataset == 'f30k':
                ckp_path = "model/checkpoint/beit3_base_patch16_384_f30k_retrieval.pth"
        elif model_size == 'large':
            if model_ptm_dataset == 'coco':
                ckp_path = "model/checkpoint/beit3_large_patch16_384_coco_retrieval.pth"
            elif model_ptm_dataset == 'f30k':
                ckp_path = "model/checkpoint/beit3_large_patch16_384_f30k_retrieval.pth"

        return ckp_path
    
    def get_transform(self, img_size):
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=3), 
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD)
        ])

        return transform
    
    def load_model_and_may_interpolate(self, ckpt_path, model, model_key, model_prefix):
        if ckpt_path.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                ckpt_path, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(ckpt_path, map_location='cpu')

        print("Load ckpt from %s" % ckpt_path)
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        # interpolate position embedding
        for pos_embed_key in ("vision_pos_embed", "pos_embed", "beit3.encoder.embed_positions.A.weight"):
            if pos_embed_key in checkpoint_model:
                pos_embed_checkpoint = checkpoint_model[pos_embed_key]
                embedding_size = pos_embed_checkpoint.shape[-1]
                if pos_embed_key == "beit3.encoder.embed_positions.A.weight":
                    # being consistent with Fairseq, which starts from 2 for position embedding
                    torchscale_model = True
                    num_patches = model.beit3.vision_embed.num_patches
                    num_extra_tokens = model.beit3.vision_embed.num_position_embeddings() + 2 - num_patches
                else:
                    torchscale_model = False
                    num_patches = model.patch_embed.num_patches
                    num_extra_tokens = getattr(model, pos_embed_key).shape[-2] - num_patches
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                    if torchscale_model:
                        extra_tokens = pos_embed_checkpoint[:num_extra_tokens].unsqueeze(0)
                        # only the position tokens are interpolated
                        pos_tokens = pos_embed_checkpoint[num_extra_tokens:]
                    else:
                        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                        # only the position tokens are interpolated
                        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
                    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    if torchscale_model:
                        new_pos_embed = new_pos_embed.squeeze(0)
                    checkpoint_model[pos_embed_key] = new_pos_embed

        self.load_state_dict(model, checkpoint_model, prefix=model_prefix)

    def load_state_dict(self, model, state_dict, prefix='', ignore_missing="relative_position_index"):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(
                prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        load(model, prefix=prefix)

        warn_missing_keys = []
        ignore_missing_keys = []
        for key in missing_keys:
            keep_flag = True
            for ignore_key in ignore_missing.split('|'):
                if ignore_key in key:
                    keep_flag = False
                    break
            if keep_flag:
                warn_missing_keys.append(key)
            else:
                ignore_missing_keys.append(key)

        missing_keys = warn_missing_keys

        if len(missing_keys) > 0:
            print("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            print("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(ignore_missing_keys) > 0:
            print("Ignored weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, ignore_missing_keys))
        if len(error_msgs) > 0:
            print('\n'.join(error_msgs))

