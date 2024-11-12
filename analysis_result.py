import json
import numpy as np


total_result_dict = {}

for seed in range(1, 11):
    file = f"LOVM/exp_result/mpnet_template_seed_{seed}.json"
    f = open(file, 'r')

    result_dict = json.load(f)

    for key, _ in result_dict.items():
        if key not in total_result_dict.keys():
            total_result_dict[key] = {"k_tau": [], "acc": [], "total": []}
        total_result_dict[key]["k_tau"].append(result_dict[key]['k_tau'][0])
        total_result_dict[key]["acc"].append(result_dict[key]['acc'][0])
        total_result_dict[key]["total"].append(result_dict[key]['k_tau'][0] + result_dict[key]['acc'][0])

result_analysis_dict = {}
for key, result_dict in total_result_dict.items():
    print(f"========================== {key} ==========================")
    result_analysis_dict[key] = {"k_tau_mean": np.mean(total_result_dict[key]['k_tau']), "k_tau_std": np.std(total_result_dict[key]['k_tau']), "acc_mean": np.mean(total_result_dict[key]['acc']), "acc_std": np.std(total_result_dict[key]['acc']), "total_mean": np.mean(total_result_dict[key]['total']), "total_std": np.std(total_result_dict[key]['total'])}
    print(result_analysis_dict[key] )