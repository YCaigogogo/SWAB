import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import pickle
import numpy as np
import ot
import yaml
import torch.nn.functional as F
import argparse


def add_prompt(cls_names, dataset):
    with open(f'LOVM/templates/{dataset}.txt', 'r') as f:
        templates = [line.rstrip('\n') for line in f.readlines()]

    template_num = len(templates)

    class_prompt_inputs = []
    for classname in cls_names:
        try:
            texts = [template.replace('{c}', classname) for template in templates]  # format with class
            class_prompt_inputs.append(texts)
        except:
            assert 0

    return class_prompt_inputs, template_num


def calc_class_relatedness(emd_model="mpnet", text_type='template', thres=0.5):
    with open("data/datasets_name.txt", 'r') as f:
        dataset_names = [line.rstrip('\n') for line in f.readlines()]

    class_name_dict = {}
    for dataset in dataset_names:
        class_name_file = f"data/datasets/classnames/{dataset}.txt"

        with open(class_name_file, 'r') as file:
            class_name_list = [line.strip() for line in file]
        
        class_name_dict[dataset] = class_name_list

    # load model
    if emd_model == "mpnet":
        model = SentenceTransformer('model/checkpoint/sentence-transformers_all-mpnet-base-v2') 
        model = model.cuda()
        model.eval()

    # =============== get class emd ===============
    class_emd = {}
    for dataset_name in tqdm(dataset_names):
        dataset_cls_list = class_name_dict[dataset_name]
        dataset_cls_prompt_list, template_num = add_prompt(dataset_cls_list, dataset_name)
        if text_type == "template":
            total_dataset_cls_prompt_list = [element for sublist in dataset_cls_prompt_list for element in sublist]
            total_dataset_tensor = model.encode(total_dataset_cls_prompt_list, convert_to_tensor=True)
            sub_tensors = total_dataset_tensor.split(template_num)
            means = [t.mean(dim=0) for t in sub_tensors]
            dataset_class_emd = torch.stack(means)

        else:
            dataset_class_emd = model.encode(dataset_cls_list, convert_to_tensor=True)
        
        class_emd[dataset_name] = dataset_class_emd
    
    f = open(f"task_relation/class_emd/{emd_model}_{text_type}.pkl", 'wb')
    pickle.dump(class_emd, f)
    # =============================================
    f = open(f"task_relation/class_emd/{emd_model}_{text_type}.pkl", 'rb')
    class_emd = pickle.load(f)
    
    def calc_filtered_class(exclude_dataset=False, dataset=None):
        filtered_src_class = {}
        for tgt_dataset in dataset_names:
            filtered_src_class[tgt_dataset] = {}
            
            tgt_dataset_class_emd = class_emd[tgt_dataset]

            for src_dataset in dataset_names:
                if exclude_dataset: 
                    flag = (src_dataset != tgt_dataset) and (src_dataset != dataset)
                else:
                    flag = (src_dataset != tgt_dataset)
                if flag:
                    src_dataset_class_emd = class_emd[src_dataset]
                    sim_mat = util.cos_sim(src_dataset_class_emd, tgt_dataset_class_emd)

                    max_values, _ = torch.max(sim_mat, dim=1)

                    selected_class_indices = [index for index, value in enumerate(max_values) if value > thres]
                    selected_class_names = [class_name_dict[src_dataset][i] for i in selected_class_indices]

                    filtered_src_class[tgt_dataset][src_dataset] = (selected_class_indices, selected_class_names)
        
        if exclude_dataset:
            file_path = f"task_relation/filtered_class/{emd_model}_{text_type}_{thres}_{dataset}.pkl"
        else:
            file_path = f"task_relation/filtered_class/{emd_model}_{text_type}_{thres}.pkl"
        f = open(file_path, 'wb')
        pickle.dump(filtered_src_class, f)

    calc_filtered_class()
    for dataset_name in dataset_names:
        calc_filtered_class(exclude_dataset=True, dataset=dataset_name)


def calc_trans_mat_with_ot(emd_model='mpnet', text_type='template', transform_k=1, m=1, thres=0.5):
    with open("data/datasets_name.txt", 'r') as f:
        dataset_names = [line.rstrip('\n') for line in f.readlines()]

    with open(f"task_relation/class_emd/{emd_model}_{text_type}.pkl", 'rb') as f:
        class_emd = pickle.load(f)

    def calc_transport_matrix(exclude_dataset=False, dataset=None, filtered_src_class=None):
        transport_matrix = {}

        for target_dataset in dataset_names:
            filtered_src_class_list = filtered_src_class[target_dataset]
            src_cls_emd_list = []
            for src_dataset, (class_indices, class_names) in filtered_src_class_list.items():
                src_cls_emd_list.append(class_emd[src_dataset][class_indices])
            
            src_cls_emd = torch.cat(src_cls_emd_list, dim=0)
            tgt_cls_emd = class_emd[target_dataset]
            
            src_cls_emd = F.normalize(src_cls_emd, p=2, dim=1)
            tgt_cls_emd = F.normalize(tgt_cls_emd, p=2, dim=1)

            # ============== optimal transport ==============
            src_cls_num = src_cls_emd.shape[0]
            tgt_cls_num = tgt_cls_emd.shape[0]

            if src_cls_num != 0:
                a = np.ones(src_cls_num) / src_cls_num
                b = np.ones(tgt_cls_num) / tgt_cls_num

                sim_matrix = src_cls_emd @ tgt_cls_emd.T
                sim_matrix = sim_matrix.cpu().numpy()

                if transform_k == -1:
                    cost_matrix = 1 - sim_matrix
                else:
                    cost_matrix = np.exp(transform_k * (1-sim_matrix))

                if m == 1:
                    T = ot.emd(a,b, cost_matrix)
                else:
                    T = ot.partial.partial_wasserstein(a,b,cost_matrix,m=m)
                
                T = torch.from_numpy(T)

                transport_matrix[target_dataset] = T
        
        if exclude_dataset:
            f = open(f"task_relation/transport_matrix/{emd_model}_{text_type}_{thres}_{transform_k}_{m}_{dataset}.pkl", 'wb')
        else:
            f = open(f"task_relation/transport_matrix/{emd_model}_{text_type}_{thres}_{transform_k}_{m}.pkl", 'wb')

        pickle.dump(transport_matrix, f)

    file_path = f"task_relation/filtered_class/{emd_model}_{text_type}_{thres}.pkl"
    f = open(file_path, 'rb')
    filtered_src_class = pickle.load(f)
    calc_transport_matrix(exclude_dataset=False, filtered_src_class=filtered_src_class)

    for dataset_name in dataset_names:
        file_path = f"task_relation/filtered_class/{emd_model}_{text_type}_{thres}_{dataset_name}.pkl"
        f = open(file_path, 'rb')
        filtered_src_class = pickle.load(f)
        calc_transport_matrix(exclude_dataset=True, dataset=dataset_name, filtered_src_class=filtered_src_class)


def calc_ptm_pred_task_acc(emd_model='mpnet', text_type='template', transform_k=1, m=1, thres=0.5):
    calc_trans_mat_with_ot(emd_model=emd_model, text_type=text_type, transform_k=transform_k, m=m, thres=thres)

    file_path = f"task_relation/transport_matrix/{emd_model}_{text_type}_{thres}_{transform_k}_{m}.pkl"
    f = open(file_path, 'rb')
    transport_matrix = pickle.load(f)

    file_path = f"task_relation/filtered_class/{emd_model}_{text_type}_{thres}.pkl"
    f = open(file_path, 'rb')
    filtered_src_class = pickle.load(f)

    with open("data/datasets_name.txt", 'r') as f:
        dataset_names = [line.rstrip('\n') for line in f.readlines()]

    with open("model/models.yml", 'r') as file:
        model_names = yaml.safe_load(file)

    model_names = [tuple(m) for m in model_names]
    model_names = ["_".join(m) for m in model_names]

    model_rank_mat_on_hist_task = {}
    pred_model_rank_mat = {}

    # get model rank on hist datasets' classes
    for tgt_dataset in tqdm(dataset_names):
        if tgt_dataset in transport_matrix:
            all_model_acc_list = []
            for model in model_names:
                one_model_acc_list = []

                with open(f"ptm_stats/stats_on_hist_task/class_level_acc/{model}.pkl", 'rb') as f:
                    model_acc_dict = pickle.load(f)

                    for src_dataset in dataset_names:
                        if src_dataset != tgt_dataset:
                            filtered_class_names = filtered_src_class[tgt_dataset][src_dataset]
                            one_model_acc_list = one_model_acc_list + [model_acc_dict[src_dataset][class_name] for class_name in filtered_class_names[1]]

                all_model_acc_list.append(one_model_acc_list)

            model_acc_mat = torch.tensor(all_model_acc_list)

            model_rank_mat = model_acc_mat.argsort(dim=0).argsort(dim=0)
            model_rank_mat_on_hist_task[tgt_dataset] = model_rank_mat.cpu().numpy()

            # get pred model rank on target datasets' classes
            pred_model_rank_mat[tgt_dataset] = model_rank_mat.float().cuda() @ transport_matrix[tgt_dataset].float().cuda()

        else:
            pred_model_rank_mat[tgt_dataset] = 0

    with open(f"ptm_stats/pred_stats_on_target_task/pred_model_rank/{emd_model}_{text_type}_{thres}_{transform_k}_{m}.pkl", 'wb') as f:
        pickle.dump(pred_model_rank_mat, f)


    # ================ calc model mean rank ================
    total_model_acc_dict = {}
    
    for tgt_dataset in tqdm(dataset_names):
        total_model_acc_list = []
        for model_name in model_names:
            total_class_level_acc_list = []
            with open(f"ptm_stats/stats_on_hist_task/class_level_acc/{model_name}.pkl", 'rb') as f:
                model_acc_dict = pickle.load(f)
                for dataset, class_acc_dict in model_acc_dict.items():
                    if dataset != tgt_dataset:
                        for class_name, acc in class_acc_dict.items():
                            total_class_level_acc_list.append(acc)
            
            total_model_acc_list.append(total_class_level_acc_list)

        model_acc_mat = torch.tensor(total_model_acc_list)
        model_rank_mat = model_acc_mat.argsort(dim=0).argsort(dim=0)

        mean_values = torch.mean(model_rank_mat.float(), dim=1)

        mean_list = mean_values.tolist()

        total_model_acc_dict[tgt_dataset] = mean_list

    with open(f"ptm_stats/pred_stats_on_target_task/pred_model_rank/model_class_rank_mean.pkl", 'wb') as f:
        pickle.dump(total_model_acc_dict, f)       
    # =====================================================
    

def calc_ptm_pred_gap_vector(emd_model='mpnet', text_type='template', transform_k=1, m=1, thres=0.5, use_dataset_level_gap_vector=False):
    calc_trans_mat_with_ot(emd_model=emd_model, text_type=text_type, transform_k=transform_k, m=m, thres=thres)
    with open("data/datasets_name.txt", 'r') as f:
        dataset_names = [line.rstrip('\n') for line in f.readlines()]

    class_name_dict = {}
    for dataset in dataset_names:
        class_name_file = f"data/datasets/classnames/{dataset}.txt"

        with open(class_name_file, 'r') as file:
            class_name_list = [line.strip() for line in file]
        
        class_name_dict[dataset] = class_name_list

    for dataset_name in tqdm(dataset_names):
        print(f"================== {dataset_name} ==================")
        file_path = f"task_relation/transport_matrix/{emd_model}_{text_type}_{thres}_{transform_k}_{m}_{dataset_name}.pkl"
        f = open(file_path, 'rb')
        transport_matrix = pickle.load(f)

        file_path = f"task_relation/filtered_class/{emd_model}_{text_type}_{thres}_{dataset_name}.pkl"
        f = open(file_path, 'rb')
        filtered_src_class = pickle.load(f)

        with open("model/models.yml", 'r') as file:
            model_names = yaml.safe_load(file)

        model_names = [tuple(m) for m in model_names]
        model_names = ["_".join(m) for m in model_names]

        model_modality_gap_on_hist_task = {}
        pred_model_modality_gap = {}

        for tgt_dataset in tqdm(dataset_names):
            for model in model_names:
                if model not in model_modality_gap_on_hist_task:
                    model_modality_gap_on_hist_task[model] = {}
                    pred_model_modality_gap[model] = {}

                one_model_modality_gap_list = []

                with open(f"ptm_stats/stats_on_hist_task/modality_gap/{model}.pkl", 'rb') as f:
                    model_modality_gap_dict = pickle.load(f)
                    for src_dataset in dataset_names:
                        if src_dataset != tgt_dataset and src_dataset != dataset_name:
                            filtered_class_names = filtered_src_class[tgt_dataset][src_dataset]
                            if use_dataset_level_gap_vector:
                                one_model_modality_gap_list = one_model_modality_gap_list + [model_modality_gap_dict[src_dataset]["dataset_mean"] for _ in filtered_class_names[1]]
                            else:
                                one_model_modality_gap_list = one_model_modality_gap_list + [model_modality_gap_dict[src_dataset][class_name] for class_name in filtered_class_names[1]]
                
                if len(one_model_modality_gap_list) != 0:
                    one_model_modality_gap_on_hist_task = torch.stack(one_model_modality_gap_list, dim=0)
                    model_modality_gap_on_hist_task[model][tgt_dataset] = one_model_modality_gap_on_hist_task
                    target_cls_num = transport_matrix[tgt_dataset].shape[1]
                    pred_model_modality_gap[model][tgt_dataset] = transport_matrix[tgt_dataset].T.float().cuda() @ one_model_modality_gap_on_hist_task.float().cuda() * target_cls_num
                
                else:
                    tgt_cls_num = len(class_name_dict[tgt_dataset])
                    feat_dim = model_modality_gap_dict['cifar100']['apple'].shape[0]
                    pred_model_modality_gap[model][tgt_dataset] = torch.zeros((tgt_cls_num, feat_dim))

        if use_dataset_level_gap_vector:
            with open(f"ptm_stats/pred_stats_on_target_task/pred_model_modality_gap/dataset_level_{emd_model}_{text_type}_{thres}_{transform_k}_{m}_{dataset_name}.pkl", 'wb') as f:
                pickle.dump(pred_model_modality_gap, f)
        else:
            with open(f"ptm_stats/pred_stats_on_target_task/pred_model_modality_gap/{emd_model}_{text_type}_{thres}_{transform_k}_{m}_{dataset_name}.pkl", 'wb') as f:
                pickle.dump(pred_model_modality_gap, f)

def get_args_parser():
    parser = argparse.ArgumentParser('SWAB', add_help=False)
    parser.add_argument('--m', default=0.9, type=float)
    parser.add_argument('--thres', default=0.5, type=float)
    parser.add_argument('--transform_k', default=1, type=int)

    return parser


def main():
    parser = argparse.ArgumentParser('modelGPT encode', parents=[get_args_parser()])
    args = parser.parse_args()  

    # print("==================== calculate class relatedness ====================")
    calc_class_relatedness(emd_model='mpnet', text_type='template', thres=args.thres)
    # print("==================== calculate ptm pred task accuracy ====================")
    calc_ptm_pred_task_acc(emd_model='mpnet', text_type='template', m=args.m, thres=args.thres, transform_k=args.transform_k)
    # print("==================== calculate ptm pred gap vector ====================")
    calc_ptm_pred_gap_vector(emd_model='mpnet', use_dataset_level_gap_vector=False)
    calc_ptm_pred_gap_vector(emd_model='mpnet', use_dataset_level_gap_vector=True)


if __name__ == "__main__":
    main()