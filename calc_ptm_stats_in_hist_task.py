import open_clip
import yaml
import torch
from data.get_dataset import get_dataset
from torchvision import transforms
from tqdm import tqdm
import os
from model.blip_class import BLIP
from model.beit_class import BEIT3
import pickle
import torch.nn.functional as F
import json


def get_model_names():
    with open("model/models.yml", 'r') as file:
        model_names = yaml.safe_load(file)

    model_names = [tuple(m) for m in model_names]

    return model_names

def get_dataset_and_class_names():
    with open("data/datasets_name.txt", 'r') as f:
        dataset_names = [line.rstrip('\n') for line in f.readlines()]

    dataset_dict = {}
    for d in dataset_names:
        _, test_dataset, _ = get_dataset(d)
        dataset_dict[d] = test_dataset

    class_name_dict = {}
    for dataset in dataset_names:
        class_name_file = f"data/datasets/classnames/{dataset}.txt"

        with open(class_name_file, 'r') as file:
            class_name_list = [line.strip() for line in file]
        
        class_name_dict[dataset] = class_name_list
    
    return dataset_names, dataset_dict, class_name_dict

def run_one_model_for_img_feat(model, model_name, dataset_names, dataset_dict, class_name_dict, preprocess):
    img_feat_dict = {}

    for d in tqdm(dataset_names,  desc=f'{model_name[0]}, {model_name[1]}', leave=False):
        dataset = dataset_dict[d]

        if d in {"fer2013","mnist"}:
            preprocess_list = [transforms.Grayscale(num_output_channels=3)]+list(preprocess.transforms)
            dataset.transform = transforms.Compose(preprocess_list)

        else:
            dataset.transform = preprocess

        # get dataloader
        dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=128,
                num_workers=8,
                pin_memory=True,
                shuffle=False,
            )
        
        with torch.no_grad():
            data = {}
            for images, labels in tqdm(dataloader, desc='Dataloader', leave=False):
                images = images.cuda()
                labels = labels.cuda()

                image_features = model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)

                for feature, label in zip(image_features.detach().cpu(), labels):
                    class_name = class_name_dict[d][label.item()]
                    if class_name not in data:
                        data[class_name] = []
                    data[class_name].append(feature)

            for class_name, feat_list in data.items():
                data[class_name] = torch.stack(feat_list)

        img_feat_dict[d] = data 

    return img_feat_dict

def calculate_text_classifier_weights(model, classnames, templates, tokenizer=None, model_name=None):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames, leave=False):
            texts = [template.replace('{c}', classname) for template in templates]  

            text_feat_list = []
            if "BEIT" in model_name:
                for text in texts:
                    text_features = model.encode_text(text)
                    text_feat_list.append(text_features)
                class_embeddings = torch.cat(text_feat_list, dim=0)
            elif "BLIP" in model_name:
                for text in texts:
                    text_features = model.encode_text(text)
                    text_feat_list.append(text_features)
                class_embeddings = torch.cat(text_feat_list, dim=0)
            else:
                texts = tokenizer(texts).cuda()
                class_embeddings = model.encode_text(texts)

            class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights

def calculate_captions_text_feat(model, datasets, tokenizer=None, model_name=None):
    with torch.no_grad():
        captions_feat = {}
        for classname, captions in tqdm(datasets.items()):
            text_feat_list = []
            try:
                if "BEIT" in model_name or "BLIP" in model_name:
                    for text in captions:
                        text_features = model.encode_text(text)
                        text_feat_list.append(text_features)
                    text_features = torch.cat(text_feat_list, dim=0)
                else:
                    texts = tokenizer(captions).cuda()
                    text_features = model.encode_text(texts)

                text_features /= text_features.norm(dim=-1, keepdim=True)

                captions_feat[classname] = text_features

            except:
                print(captions)
                assert 0
    
    return captions_feat
    
def calculate_syn_text_feat(model, datasets, tokenizer=None, model_name=None):
    with torch.no_grad():
        syns_feat = {}
        for classname, syns in tqdm(datasets.items()):
            text_feat_list = []
            try:
                if "BEIT" in model_name or "BLIP" in model_name:
                    for text in syns:
                        text_features = model.encode_text(text)
                        text_feat_list.append(text_features)
                    text_features = torch.cat(text_feat_list, dim=0)
                else:
                    texts = tokenizer(syns).cuda()
                    text_features = model.encode_text(texts)

                text_features /= text_features.norm(dim=-1, keepdim=True)

                syns_feat[classname] = text_features

            except:
                print(syns)
                assert 0
    
    return syns_feat

def run_one_model_for_text_feat(model, model_name, dataset_names, tokenizer):
    text_classifier_dict = {}
    caption_text_feat_dict = {}
    syn_text_feat_dict = {}

    caption_datasets = {}
    syn_datasets = {}
    for d in dataset_names:
        with open(f'data/captions_dataset/{d}.json', 'r') as f:
            caption_datasets[d] = json.load(f)
        with open(f'data/syn_dataset/{d}.json', 'r') as f:
            syn_datasets[d] = json.load(f)

    for dataset in tqdm(dataset_names, desc=str(model_name)):
        with open(f'LOVM/templates/{dataset}.txt', 'r') as f:
            templates = [line.rstrip('\n') for line in f.readlines()]

        with open(f'data/datasets/classnames/{dataset}.txt', 'r') as f:
            classes = [line.rstrip('\n') for line in f.readlines()]

        if "BEIT" in model_name[0] or "BLIP" in model_name[0]:
            text_classifier_dict[dataset] = calculate_text_classifier_weights(model, classes, templates, model_name=model_name[0])
            caption_text_feat_dict[dataset] = calculate_captions_text_feat(model, caption_datasets[dataset], model_name=model_name[0])
            syn_text_feat_dict[dataset] = calculate_syn_text_feat(model, syn_datasets[dataset], model_name=model_name[0])
        else:
            text_classifier_dict[dataset] = calculate_text_classifier_weights(model, classes, templates, tokenizer=tokenizer, model_name=model_name[0])
            caption_text_feat_dict[dataset] = calculate_captions_text_feat(model, caption_datasets[dataset], tokenizer=tokenizer, model_name=model_name[0])
            syn_text_feat_dict[dataset] = calculate_syn_text_feat(model, syn_datasets[dataset], model_name=model_name[0], tokenizer=tokenizer)

    return text_classifier_dict, caption_text_feat_dict, syn_text_feat_dict

def extract_img_feat_and_text_feat(model_names, dataset_names, dataset_dict, class_name_dict):
    for m in tqdm(model_names, desc="model"):
        model_name = "_".join(m)

        print(f"--------------- model {m} ---------------")
        if os.path.exists(f'ptm_stats/stats_on_hist_task/img_feat/{model_name}.pkl') and os.path.exists(f'ptm_stats/stats_on_hist_task/text_classifier/{model_name}.pkl') and os.path.exists(f'ptm_stats/stats_on_hist_task/caption_text_feat/{model_name}.pkl') and os.path.exists(f'ptm_stats/stats_on_hist_task/syn_text_feat/{model_name}.pkl'):
            continue
        else:
            if "BEIT" in m[0]:
                model_type, model_size, model_ptm_dataset = m[1].split("_")
                model = BEIT3(model_size=model_size, model_ptm_dataset=model_ptm_dataset)
                preprocess = model.transform
                model.model.to('cuda')
                model.model.eval()
                tokenizer = None
            elif "BLIP" in m[0]:
                model_type, model_size, model_ptm_dataset = m[1].split("_")
                model = BLIP(model_size=model_size, model_ptm_dataset=model_ptm_dataset)
                preprocess = model.transform
                model.model.to('cuda')
                model.model.eval()
                tokenizer = None
            else:
                model, _, preprocess = open_clip.create_model_and_transforms(m[0], m[1], cache_dir='model/checkpoint') # TODO
                model.to('cuda')
                model.eval()
                tokenizer = open_clip.get_tokenizer(m[0])

        img_feat_dict = run_one_model_for_img_feat(model, m, dataset_names, dataset_dict, class_name_dict, preprocess) # return dict

        with open(f'ptm_stats/stats_on_hist_task/img_feat/{model_name}.pkl', 'wb') as file:
            pickle.dump(img_feat_dict, file)

        text_classifier_dict, caption_text_feat_dict, syn_text_feat_dict = run_one_model_for_text_feat(model, m, dataset_names, tokenizer)

        with open(f'ptm_stats/stats_on_hist_task/text_classifier/{model_name}.pkl', 'wb') as file:
            pickle.dump(text_classifier_dict, file)

        with open(f'ptm_stats/stats_on_hist_task/caption_text_feat/{model_name}.pkl', 'wb') as file:
            pickle.dump(caption_text_feat_dict, file)
        
        with open(f'ptm_stats/stats_on_hist_task/syn_text_feat/{model_name}.pkl', 'wb') as file:
            pickle.dump(syn_text_feat_dict, file)

def cal_gap_dict(model_names, dataset_names, class_name_dict):
    for m in tqdm(model_names, desc="model"):
        model_name = "_".join(m)

        modality_gap_path = f"ptm_stats/stats_on_hist_task/modality_gap/{model_name}.pkl"
        f = open(modality_gap_path, 'wb')
        modality_gap_dict = {}

        img_feat_path = f"ptm_stats/stats_on_hist_task/img_feat/{model_name}.pkl"
        f1 = open(img_feat_path, 'rb')
        image_data_dict = pickle.load(f1)

        text_classifier_path = f"ptm_stats/stats_on_hist_task/text_classifier/{model_name}.pkl"
        f2 = open(text_classifier_path, 'rb')
        text_classifier_dict = pickle.load(f2)

        caption_text_feat_path = f'ptm_stats/stats_on_hist_task/caption_text_feat/{model_name}.pkl'
        f3 = open(caption_text_feat_path, 'rb')
        caption_text_data_dict = pickle.load(f3)

        for dataset_name in tqdm(dataset_names):
            modality_gap_dict[dataset_name] = {}
            total_modality_gap = []

            class_name_list = class_name_dict[dataset_name]

            img_data_list = []
            for _, img_data in image_data_dict[dataset_name].items():
                img_data_list.append(img_data)

            total_img_data = torch.cat(img_data_list, dim=0)

            caption_text_data_list = []
            for _, text_data in caption_text_data_dict[dataset_name].items():
                caption_text_data_list.append(text_data)

            total_caption_text_data = torch.cat(caption_text_data_list, dim=0)

            text_center = torch.mean(total_caption_text_data, dim=0)
            img_center = torch.mean(total_img_data, dim=0)

            text_std = torch.std(total_caption_text_data, dim=0)
            img_std = torch.std(total_img_data, dim=0)

            text_classifiers = text_classifier_dict[dataset_name]

            text_classifiers = ((text_classifiers.T.cuda() - text_center.cuda()) / text_std.cuda()).T

            for i, class_name in enumerate(class_name_list):
                img_feat = image_data_dict[dataset_name][class_name]
                img_feat = (img_feat - img_center) / img_std
                text_classifier = text_classifiers[:, i]
                modality_gaps = img_feat.cuda() - text_classifier.cuda()
                total_modality_gap.append(modality_gaps)
                modality_gap_dict[dataset_name][class_name] = torch.mean(modality_gaps, dim=0)
            
            modality_gap_dict[dataset_name]['dataset_mean'] = torch.mean(torch.cat(total_modality_gap, dim=0), dim=0)

        pickle.dump(modality_gap_dict, f)


def cal_model_acc(model_names, dataset_names, class_name_dict):
    for m in tqdm(model_names, desc="model"):
        model_performance_list = []

        model_name = "_".join(m)

        print("model_name:", model_name)

        acc_path = f"ptm_stats/stats_on_hist_task/class_level_acc/{model_name}.pkl"
        f = open(acc_path, 'wb')
        acc_dict = {}

        img_feat_path = f"ptm_stats/stats_on_hist_task/img_feat/{model_name}.pkl"
        f1 = open(img_feat_path, 'rb')
        image_data_dict = pickle.load(f1)

        text_classifier_path = f"ptm_stats/stats_on_hist_task/text_classifier/{model_name}.pkl"
        f2 = open(text_classifier_path, 'rb')
        text_classifier_dict = pickle.load(f2)

        for dataset_name in dataset_names:
            total_top1_correct = 0
            total_top5_correct = 0
            total_ins_num = 0

            acc_dict[dataset_name] = {}
            class_name_list = class_name_dict[dataset_name]

            zs_classifier = text_classifier_dict[dataset_name].cuda()

            for i, class_name in enumerate(class_name_list):
                data = image_data_dict[dataset_name][class_name].cuda()
                pred_logits = data @ zs_classifier
                preds = torch.argmax(pred_logits, dim=1)
                labels = torch.full_like(preds, i) 
                top1_correct = (preds == labels).sum().item()

                # Calculate the class-level accuracy
                accuracy = top1_correct / len(labels)

                acc_dict[dataset_name][class_name] = accuracy

                # Calculate the top 1 acc
                total_top1_correct += top1_correct
                total_ins_num += len(labels)

                # Calculate the top 5 acc
                if int(pred_logits.shape[1]) > 5:
                    top5_preds = torch.topk(pred_logits, k=5, dim=1).indices 
                    top5_correct = top5_preds.eq(labels.view(-1, 1)).sum().item()
                else:
                    top5_correct = len(labels)
                total_top5_correct += top5_correct

            top1_acc = total_top1_correct / total_ins_num
            top5_acc = total_top5_correct / total_ins_num

            model_performance_list.append((dataset_name, " ".join(m), m[0], m[1], top1_acc, top5_acc))

        pickle.dump(acc_dict, f)

        csv_filename = "LOVM/eval_table.csv"
        file_exists = os.path.isfile(csv_filename)

        # ================= save eval_table.csv =================
        # with open(csv_filename, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     if not file_exists:
        #         writer.writerow(["dataset", "model_fullname", "model","pretrained","acc1","acc5"])
        #     writer.writerows(model_performance_list)


def main():
    model_names = get_model_names()

    dataset_names, dataset_dict, class_name_dict = get_dataset_and_class_names()

    # calculate ptm's class-level gap vector
    print("==================== extract img feat and text feat ====================")
    extract_img_feat_and_text_feat(model_names, dataset_names, dataset_dict, class_name_dict)
    
    print("==================== calculate modality gap vector ====================")
    cal_gap_dict(model_names, dataset_names, class_name_dict)

    # calculate ptm's class-level accuracy
    print("==================== calculate class level accuracy ====================")
    cal_model_acc(model_names, dataset_names, class_name_dict)


if __name__ == "__main__":
    main()