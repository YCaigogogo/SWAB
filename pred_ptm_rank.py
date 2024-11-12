import argparse
from sklearn.metrics import r2_score, f1_score, accuracy_score
import pandas as pd
import yaml
from tqdm import tqdm
import pickle
import torch
import random
import numpy as np
import torch.nn.functional as F
from LOVM.modelGPT.constants import MODELS, ALL_FEATURES, FEATURES_SET
from collections import defaultdict
from typing import Iterable, Union
from itertools import chain, combinations
from LOVM.lovm import LOVM
from LOVM.modelGPT.model_gpt_predictor import ModelGPTPredictor
import json
import os


def get_args_parser():
    parser = argparse.ArgumentParser('SWAB', add_help=False)
    parser.add_argument('--m', default=0.9, type=float)
    parser.add_argument('--thres', default=0.5, type=float)
    parser.add_argument('--transform_k', default=1, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--text_model', default='mpnet', type=str)
    parser.add_argument('--text_type', default='template', type=str)
    parser.add_argument('--model', default='linear_regression', type=str)
    parser.add_argument('--modelgpt_use_stats', default='C', type=str)
    parser.add_argument(
        "--no_subsets",
        action="store_false",
        default=True,
    ) 
    parser.add_argument(
        "--use_pred_ranking_1",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_pred_ranking_2",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_model_avg_rank",
        action="store_true",
        default=False,
    )

    return parser

def set_seed(seed):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  

def eval_on_dataset(dataset, zero_shot_model, sigma):
    pred = []
    for i in range(len(dataset)):
        data = (dataset[i].cuda() + torch.randn(dataset[i].shape).cuda() * sigma).to(zero_shot_model.device)
        pred.append(np.argmax((data @ zero_shot_model).cpu().numpy(), axis=1))

    pred = np.concatenate(pred)
    y_true = np.concatenate([i * np.ones(data.shape[0], dtype=np.int32) for i, data in enumerate(dataset)])
    return f1_score(y_true, pred, average='macro'), accuracy_score(y_true, pred)

def calc_data_metrics(data, zeroshot_weights):
    inter_close = []
    intra_close = []
    for i in range(len(data)):
        d = F.normalize(data[i], dim=1).to(zeroshot_weights.device)
        cos_sim = (d @ zeroshot_weights)
        inter_close.append(cos_sim[:, i].mean().cpu().numpy())
        intra_close.append(cos_sim[:, [ii for ii in range(len(data)) if ii != i]].mean(dim=0).max().cpu().numpy())

    cosine_sim_matrix = (zeroshot_weights.T @ zeroshot_weights)
    cosine_sim_matrix.fill_diagonal_(0)
    max_per_class, _ = cosine_sim_matrix.max(dim=0)
    return float(max_per_class.mean()), np.mean(inter_close), np.mean(intra_close)

def calculate_synonym_simularity(syn_data, model):
    syn_weights = []
    for i, v in enumerate(syn_data):
        syn = v.cuda()
        syn_weights.append((model[:, i] @ syn.T).mean().detach().cpu().numpy())
    return np.mean(syn_weights)

def load_text_data(dataset, model_names):
    captions_data = {}
    syn_data = {}
    text_classifier = {}

    for model_name in model_names:
        m = "_".join(model_name)

        with open(f"ptm_stats/stats_on_hist_task/caption_text_feat/{m}.pkl", 'rb') as f:
            caption_text_feat = pickle.load(f)
            caption_feat_list = []
            
            for class_name, caption_feat in caption_text_feat[dataset].items():
                caption_feat_list.append(caption_feat)

            captions_data[m] = caption_feat_list
        
        with open(f"ptm_stats/stats_on_hist_task/syn_text_feat/{m}.pkl", 'rb') as f:
            syn_text_feat = pickle.load(f)
            syn_feat_list = []
            
            for class_name, syn_feat in syn_text_feat[dataset].items():
                syn_feat_list.append(syn_feat)

            syn_data[m] = syn_feat_list

        text_classifier[m] = {}
        with open(f"ptm_stats/stats_on_hist_task/text_classifier/{m}.pkl", 'rb') as f:
            text_cls = pickle.load(f)
            text_classifier[m] = text_cls[dataset]

    return captions_data, syn_data, text_classifier

def add_gap(datas, modality_gap_vectors):
    datas_after_gap_bridging = []
    modality_gap_vectors_list = modality_gap_vectors.tolist()

    for data, gap_vector in zip(datas, modality_gap_vectors_list):
        data = data.cuda() + torch.tensor(gap_vector).cuda()
        datas_after_gap_bridging.append(data.detach().cpu())
    
    return datas_after_gap_bridging

def calc_model_stats_on_hist_task_text_sample(args):
    with open("model/models.yml", 'r') as file:
        model_names = yaml.safe_load(file)

    model_names = [tuple(m) for m in model_names]

    with open("data/datasets_name.txt", 'r') as f:
        dataset_names = [line.rstrip('\n') for line in f.readlines()]

    sigma = 0.15

    for dataset_name in tqdm(dataset_names):
        print(f"================== {dataset_name} ==================")
        eval_table = pd.read_csv("LOVM/eval_table.csv")

        eval_table_metrics = f"LOVM/SWAB_inputs/{args.text_model}_{args.text_type}_target_dataset_{dataset_name}_seed_{args.seed}.csv"

        if os.path.exists(eval_table_metrics):
            continue
        
        gaps_file_path = f"ptm_stats/pred_stats_on_target_task/pred_model_modality_gap/{args.text_model}_{args.text_type}_0.5_1_1_{dataset_name}.pkl"

        with open(gaps_file_path, 'rb') as file:
            gaps_dict = pickle.load(file)

        for dataset_name_2 in tqdm(dataset_names):
            captions_datas, syn_datas, text_classifiers = load_text_data(dataset=dataset_name_2, model_names=model_names)

            for m in model_names:
                in_tmp=eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset=="imagenet1k")]
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'IN-score']=float(in_tmp.acc1.iloc[0])
                model_name = "_".join(m)
                total_text_data = torch.cat(captions_datas[model_name])
                text_center = torch.mean(total_text_data, dim=0)
                text_std = torch.std(total_text_data, dim=0)

                z_score_text_classifiers = ((text_classifiers[model_name].T.cuda() - text_center.cuda()) / text_std.cuda()).T

                z_score_captions_data = []
                for caption_d in captions_datas[model_name]:
                    z_score_captions_data.append((caption_d.cuda() - text_center.cuda()) / text_std.cuda())

                z_score_add_gap_captions_data = add_gap(z_score_captions_data, gaps_dict[model_name][dataset_name_2])

                sigma = 0.15

                ## z-score
                tmp=eval_on_dataset(z_score_add_gap_captions_data, z_score_text_classifiers, sigma)
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'z-gap-text-f1']=tmp[0]
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'z-gap-text-acc1']=tmp[1]

                max_intraclass, inter_close, intra_close = calc_data_metrics(z_score_add_gap_captions_data, z_score_text_classifiers)
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'z-gap-intraclass_sim']=float(max_intraclass)
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'z-gap-inter_close']=inter_close
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'z-gap-intra_close']=intra_close

                # =============================
                tmp=eval_on_dataset(captions_datas[model_name], text_classifiers[model_name], sigma)
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'text-f1']=tmp[0]
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'text-acc1']=tmp[1]

                max_intraclass, inter_close, intra_close = calc_data_metrics(captions_datas[model_name], text_classifiers[model_name])
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'intraclass_sim']=float(max_intraclass)
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'inter_close']=inter_close
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'intra_close']=intra_close

                superclass_sim = calculate_synonym_simularity(syn_datas[model_name], text_classifiers[model_name])
                eval_table.loc[(eval_table.model_fullname==m[0]+" "+m[1])&(eval_table.dataset==dataset_name_2), 'superclass_metric']=superclass_sim

        eval_table.dropna(subset=['IN-score'], inplace=True)
        eval_table.to_csv(eval_table_metrics)

def create_all_subsets(ss):
    all_sets = list(chain(*map(lambda x: combinations(ss, x), range(0, len(ss) + 1))))
    return [list(s) for s in all_sets if len(list(s)) > 0]

def run_ablation(
        features_set: Iterable[str] = ALL_FEATURES,
        model_set: Union[Iterable[str], str] = 'linear_regression',
        ablate_subset: bool = True,
        text_model="mpnet",
        text_type="template",
        seed=1,
        m=0.8,
        thres=0.5,
        transform_k=1,
        use_pred_ranking_1=True,
        use_pred_ranking_2=True,
        use_model_avg_rank=False,
) -> pd.DataFrame:

    # dict to store results
    full_results = []
    results = defaultdict(list)

    # store model type in list if there is only one model type
    if type(model_set) == str:
        model_set = [model_set]

    # create all feature combinations for ablation
    if ablate_subset:
        all_subsets = create_all_subsets(features_set)
    else:
        all_subsets = [features_set]

    # loop through all models to ablate
    for model_type in tqdm(model_set, total=len(model_set)):

        # select model and get grid search parameters
        model = MODELS[model_type]

        # loop through all feature combinations
        for ss in tqdm(all_subsets, total=len(all_subsets), desc=model_type):

            # get all models and datasets
            model_gpt = ModelGPTPredictor(
                seed=seed, features=ss, model=model, text_model=text_model, text_type=text_type, m=m, thres=thres, transform_k=transform_k, use_pred_ranking_1=use_pred_ranking_1, use_pred_ranking_2=use_pred_ranking_2, use_model_avg_rank=use_model_avg_rank)
            
            lovm = LOVM(pred_target="acc1")

            if len(ss) == 1 and ss[0] == 'IN-score':
                pred = lovm.get_imagenet_model_rank()
                best_param = None
            else:
                pred, best_param = model_gpt.loo_model_rank()

            metric = lovm.evaluate_model_rank(pred)
            results['acc'].append(metric.loc['mean', 'acc'])
            results['k_tau'].append(metric.loc['mean', 'k_tau'])

            full_results.append(metric)
            results['features'].append(ss)
            results['model'].append(model_type)
            results['best_param'].append(best_param)

    # aggregate results to dataframe
    results_df = pd.DataFrame.from_dict(results)
    results_df = round(results_df, 3)

    results_df['num_features'] = results_df.features.apply(lambda x: len(x))
    return results_df, full_results


def print_result(results_df, feat, text_model="mpnet", text_type="template", seed=1, model='linear_regression', use_pred_ranking_1=True, use_pred_ranking_2=False, use_model_avg_rank=False):
    results_df['acc_k_tau_sum'] = results_df['acc'] + results_df['k_tau']

    max_acc_row = results_df[results_df['acc_k_tau_sum'] == results_df['acc_k_tau_sum'].max()]

    # If there are multiple rows with the same maximum value,
    # select the one with the maximum "k_tau" value
    if len(max_acc_row) > 1:
        max_acc_row = max_acc_row[max_acc_row['k_tau'] == max_acc_row['k_tau'].max()]

    max_acc_row['features_set'] = '+'.join(feat.split(','))

    # ============================
    features_set = max_acc_row["features_set"].iloc[0]
    k_tau = float(max_acc_row["k_tau"].iloc[0])
    acc = float(max_acc_row["acc"].iloc[0])
    features = max_acc_row["features"].iloc[0]

    print(max_acc_row)

    file_path = f"LOVM/exp_result/{text_model}_{text_type}_seed_{seed}.json"

    if use_pred_ranking_1 and use_pred_ranking_2:
        features_set = "pred_1" + "(" + features_set + ")" + "+" + "pred_2" + "+" + model
    elif use_pred_ranking_1:
        features_set = "pred_1" + "(" + features_set + ")" + "+" + model
    elif use_pred_ranking_2:
        if use_model_avg_rank:
            features_set = "pred_2(use_model_avg_rank)"
        else:
            features_set = "pred_2"

    if not os.path.exists(file_path):
        result_dict = {}
        result_dict[features_set] = {}
        result_dict[features_set]["k_tau"] = [k_tau]
        result_dict[features_set]["acc"] = [acc]
        result_dict[features_set]["use_features"] = [features]
        result_dict[features_set]["model"] = [model]

        f = open(file_path, 'w')
        json.dump(result_dict, f)
    else:
        f = open(file_path, 'r')
        result_dict = json.load(f)
        if features_set not in result_dict:
            result_dict[features_set] = {}
            result_dict[features_set] = {}
            result_dict[features_set]["k_tau"] = [k_tau]
            result_dict[features_set]["acc"] = [acc]
            result_dict[features_set]["use_features"] = [features]
        else:
            result_dict[features_set]["k_tau"].append(k_tau)
            result_dict[features_set]["acc"].append(acc)
            result_dict[features_set]["use_features"].append(features)
    
        f = open(file_path, 'w')
        json.dump(result_dict, f)

def gen_latex(list):
    return " \hspace{-0.4em} & \hspace{-0.9em} ".join([str(round(f, 2)) for f in list[:-1]] + [str(round(list[-1],3))])

def calc_ptm_pred_rank(args):
    if args.model is None:
        model_set = MODELS.keys()
    else:
        model_set = [args.model]

    if args.modelgpt_use_stats is not None:
        features_set = []
        for f in args.modelgpt_use_stats.split(','):
            features_set += FEATURES_SET[f]
    else:
        features_set = ALL_FEATURES

    results_df, full_results = run_ablation(
        features_set=features_set,
        model_set=model_set,
        ablate_subset=args.no_subsets,
        text_model=args.text_model,
        text_type=args.text_type,
        seed=args.seed,
        m=args.m,
        thres=args.thres,
        transform_k=args.transform_k,
        use_pred_ranking_1=args.use_pred_ranking_1,
        use_pred_ranking_2=args.use_pred_ranking_2,
        use_model_avg_rank=args.use_model_avg_rank
    )


    print_result(results_df, args.modelgpt_use_stats, seed=args.seed, use_pred_ranking_1=args.use_pred_ranking_1, use_pred_ranking_2=args.use_pred_ranking_2, use_model_avg_rank=args.use_model_avg_rank, model=args.model)


def main():
    parser = argparse.ArgumentParser('modelGPT encode', parents=[get_args_parser()])
    args = parser.parse_args()    
    set_seed(args.seed)
    calc_model_stats_on_hist_task_text_sample(args)
    calc_ptm_pred_rank(args)

if __name__ == "__main__":
    main()