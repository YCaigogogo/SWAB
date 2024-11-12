# Extract statistics of the pre-trained VLMs on the open-source dataset.
python calc_ptm_stats_in_hist_task.py

# SWAB uses optimal transport to predict the statistics of pre-trained VLMs on target tasks
python pred_ptm_stats_in_target_task_with_ot.py

# Evaluate the performance of different model selection methods.
bash eval.sh

# Analyze experimental results
python analysis_result.py