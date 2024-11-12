import pandas as pd
import numpy as np
from collections import defaultdict
from scipy.stats import kendalltau
from typing import Iterable, Callable
from LOVM.constants import NUM_RANK, GROUND_TRUTH_CSV, MODEL_NAME_COL, DATASET_COL


def get_acc(pred, true, num_rank):
    pred = pred[:num_rank]
    true = true[:num_rank]
    return np.sum(
        [i in pred.index for i in true.index]
    )/len(true)

def get_k_tau(pred, true, num_rank):
    pred = pred[:num_rank]
    true = true[:num_rank]
    true = true.rename_axis('index').reset_index()
    pred = pred.rename_axis('index').reset_index()
    col_name = true.columns[-1]

    merge = pd.merge(true, pred, on='index', suffixes=('_true', '_pred'))

    true = merge[f"{col_name}_true"].rank(ascending=False).astype(int)
    pred = merge[f"{col_name}_pred"].rank(ascending=False).astype(int)

    if len(true) > 2:
        return kendalltau(true, pred).correlation
    return 0

METRIC_DICT = {
    'acc': get_acc,
    'k_tau': get_k_tau
}


class LOVM:

    def __init__(self, pred_target:str="acc1", num_rank:int = NUM_RANK, return_mean:bool = True, dataset_to_remove=[], model_to_remove=[]):

        self.num_rank = num_rank
        self.return_mean = return_mean
        self.gt_df = pd.read_csv(GROUND_TRUTH_CSV)

        self.imagenet_df = self.gt_df[~self.gt_df[DATASET_COL].isin([d for d in dataset_to_remove if d != 'imagenet1k'])]
        self.imagenet_df = self.imagenet_df[~self.imagenet_df[MODEL_NAME_COL].isin(model_to_remove)]
        self.gt_df = self.gt_df[~self.gt_df[DATASET_COL].isin(dataset_to_remove)]
        self.gt_df = self.gt_df[~self.gt_df[MODEL_NAME_COL].isin(model_to_remove)]

        self.dataset_rank_gt_df = pd.pivot_table(
            self.gt_df, values=pred_target, index=[DATASET_COL],
            columns=MODEL_NAME_COL
        )
        self.model_rank_gt_df = pd.pivot_table(
            self.gt_df, values=pred_target, index=[MODEL_NAME_COL],
            columns=DATASET_COL
        )
        self.imagenet_df = pd.pivot_table(
            self.imagenet_df, values=pred_target, index=[MODEL_NAME_COL],
            columns=DATASET_COL
        )


    def _evaluate(
            self, pred_df:pd.DataFrame, target_df:pd.DataFrame,
            eval_func_names:Iterable[Callable]
        )->pd.DataFrame:

        result_dict = defaultdict(list)
        for metric in eval_func_names:
            for c in pred_df.columns:
                pred = pred_df.sort_values(c, ascending=False)[c]
                target = target_df.sort_values(c, ascending=False)[c]
                result_dict[metric].append(METRIC_DICT[metric](pred, target, self.num_rank))

        # convert to dataframe
        result_dict['cols'] = list(pred_df.columns)
        result_df = pd.DataFrame(result_dict)
        result_df = result_df.set_index('cols').sort_index()

        # return mean or full dataframe
        if self.return_mean:
            result_df.loc['mean'] = result_df.mean()

        return result_df


    def evaluate_model_rank(
            self, model_rank_pred_df:pd.DataFrame
        )->pd.DataFrame:
        """Wrappper function around self._evalate to evalaute a dataframe of
        model ranks against a dataframe of true model ranks.
        The evaluation functions are specified by the eval_func_names argument.

        Args:
            pred_df (pd.DataFrame): dataframe of predictions
            target_df (pd.DataFrame): dataframe of targets
            eval_func_names (Iterable[Callable]): list of evaluation functions

        Returns:
            pd.DataFrame: dataframe of evaluation results
        """

        return self._evaluate(
            model_rank_pred_df, self.model_rank_gt_df, ['acc', 'k_tau']
        )

    def get_imagenet_model_rank(self) -> pd.DataFrame:
        """Get the imagenet model rank from the ground truth dataframe.

        Returns:
            pd.DataFrame: dataframe of imagenet model rank
        """

        imagenet_pred = self.imagenet_df.copy()
        for c in imagenet_pred.columns:
            imagenet_pred[c] = self.imagenet_df['imagenet1k']
        return imagenet_pred


def main():
    from modelGPT.model_gpt_predictor import ModelGPTPredictor
    from modelGPT.constants import FEATURES_CSV

    df_features = pd.read_csv(FEATURES_CSV)
    lovm = LOVM()
    imagenet_baseline = lovm.get_imagenet_model_rank()
    model_gpt = ModelGPTPredictor(df_features)
    model_rank_rank, best_params = model_gpt.loo_model_rank()
    df_eval = lovm.evaluate_model_rank(model_rank_rank)
    print(gen_latex(df_eval.acc.tolist()))
    print(gen_latex(df_eval.k_tau.tolist()))


def gen_latex(list):
    return " \hspace{-0.4em} & \hspace{-0.9em} ".join([str(round(f, 2)) for f in list[:-1]] + [str(round(list[-1],3))])

if __name__ == "__main__":
    main()
