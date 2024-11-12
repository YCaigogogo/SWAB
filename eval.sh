text_model="mpnet"
text_type="template"
gpu=0


for seed in {1..10}
do
    # INB
    CUDA_VISIBLE_DEVICES=$gpu python pred_ptm_rank.py --seed $seed --model linear_regression --text_model $text_model --text_type $text_type --modelgpt_use_stats "INB" 
    # Model AVG Rank
    CUDA_VISIBLE_DEVICES=$gpu python pred_ptm_rank.py --seed $seed --model linear_regression --text_model $text_model --text_type $text_type --use_pred_ranking_2 --use_model_avg_rank
    # ModelGPT + No INB
    CUDA_VISIBLE_DEVICES=$gpu python pred_ptm_rank.py --seed $seed --model linear_regression --text_model $text_model --text_type $text_type --modelgpt_use_stats "C,G" --thres 0.5 --use_pred_ranking_1
    # SWAB-M + No INB
    CUDA_VISIBLE_DEVICES=$gpu python pred_ptm_rank.py --seed $seed --model linear_regression --text_model $text_model --text_type $text_type --modelgpt_use_stats "GAP_C,GAP_G" --thres 0.5 --use_pred_ranking_1
    # ModelGPT
    CUDA_VISIBLE_DEVICES=$gpu python pred_ptm_rank.py --seed $seed --model linear_regression --text_model $text_model --text_type $text_type --modelgpt_use_stats "INB,C,G" --thres 0.5 --m 0.9 --use_pred_ranking_1 
    # SWAB-M
    CUDA_VISIBLE_DEVICES=$gpu python pred_ptm_rank.py --seed $seed --model linear_regression --text_model $text_model --text_type $text_type --modelgpt_use_stats "INB,GAP_C,GAP_G" --thres 0.5 --m 0.9 --use_pred_ranking_1 
    # SWAB-C
    CUDA_VISIBLE_DEVICES=$gpu python pred_ptm_rank.py --seed $seed --model linear_regression --text_model $text_model --text_type $text_type --modelgpt_use_stats "INB,C,G" --thres 0.5 --m 0.9 --use_pred_ranking_1 --use_pred_ranking_2
    # SWAB
    CUDA_VISIBLE_DEVICES=$gpu python pred_ptm_rank.py --seed $seed --model linear_regression --text_model $text_model --text_type $text_type --modelgpt_use_stats "INB,GAP_C,GAP_G" --thres 0.5 --m 0.9 --use_pred_ranking_1 --use_pred_ranking_2 
done