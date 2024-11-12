from typing import Iterable, Type, Tuple, Callable
import torch

import pandas as pd 
import numpy as np

import pickle

from LOVM.modelGPT.constants import ALL_FEATURES, FEATURES_CSV, MODELS
from LOVM.constants import MODEL_NAME_COL, DATASET_COL


replace_feature = {
    "z-gap-text-f1": 'text-f1',
    'z-gap-text-acc1': 'text-acc1',
    'z-gap-intraclass_sim': 'intraclass_sim',
    'z-gap-inter_close': 'inter_close',
    'z-gap-intra_close': 'intra_close'
}


class ModelGPTPredictor:
    
    def __init__(
            self, 
            seed = 1,
            features:Iterable[str] = ALL_FEATURES,
            model:Type = MODELS['huber'],
            model_name_col:str = MODEL_NAME_COL,
            dataset_col:str = DATASET_COL,
            text_model:str = "mpnet",
            text_type:str = "template",
            m:float = 0.8,
            thres:float = 0.5,
            transform_k:int = 1,
            use_pred_ranking_1:bool = True,
            use_pred_ranking_2:bool = True,
            use_model_avg_rank: bool = False,
        ):

        self.seed = seed
        self.features = features
        self.model = model
        self.model_name_col = model_name_col
        self.dataset_col = dataset_col
        self.pred_target = 'acc1'
        self.text_model = text_model
        self.text_type = text_type
        self.use_pred_ranking_1 = use_pred_ranking_1
        self.use_pred_ranking_2 = use_pred_ranking_2
        self.use_model_avg_rank = use_model_avg_rank
        
        with open("data/datasets_name.txt", 'r') as f:
            self.all_datasets = [line.rstrip('\n') for line in f.readlines()]

        self.load_pred_ranking_2(model=text_model, type=text_type, thres=thres, m=m, transform_k=transform_k, use_model_avg_rank=use_model_avg_rank)


    def fit_predict_model(
            self, x_train:pd.DataFrame, x_test:pd.DataFrame, 
            y_train:pd.DataFrame, 
        ) -> Tuple[np.ndarray, dict]:

        # fit and predict model
        self.model.fit(x_train, y_train)
        best_params = None
        model_pred = self.model.predict(x_test)

        return model_pred, best_params   


    def load_pred_ranking_2(self, model="mpnet", type="template", thres="0.5", transform_k=1, m=0.9, use_model_avg_rank=False):
        if use_model_avg_rank:
            path = "ptm_stats/pred_stats_on_target_task/pred_model_rank/model_class_rank_mean.pkl"
        else:
            path = f"ptm_stats/pred_stats_on_target_task/pred_model_rank/{model}_{type}_{thres}_{transform_k}_{m}.pkl"
        f = open(path, 'rb')

        self.pred_ranking_2 = pickle.load(f)
    
    def performance_to_rank(self, performance_list):
        return np.argsort(np.argsort([-x for x in performance_list]))

    def borda_count(self, performance_lists, weight_list):
        rank_lists = [self.performance_to_rank(performance_list) for performance_list in performance_lists]

        num_models = len(rank_lists[0])
        scores = [0] * num_models

        for idx, rank_list in enumerate(rank_lists):
            weight = weight_list[idx]
            for model_id, rank in enumerate(rank_list):
                scores[model_id] += (num_models - rank) * weight

        return scores

    def _predict_target(
            self, df, split_col:str, split_val:str, target_type:str, pred_target:str
        ) -> Tuple[pd.DataFrame, dict]:

        if self.use_pred_ranking_1:
            if isinstance(self.pred_ranking_2[split_val], int) and self.pred_ranking_2[split_val] == 0:
                features = [replace_feature[item] if item in replace_feature else item for item in self.features]
            else:
                features = self.features
            
            x_train = df.loc[df[split_col] != split_val][features]
            x_test = df.loc[df[split_col] == split_val][features]
            y_train = df.loc[df[split_col] != split_val][pred_target].values.ravel()

            # fit predict model
            model_pred, best_params = self.fit_predict_model(x_train, x_test, y_train) 

            if self.use_pred_ranking_2:
                if isinstance(self.pred_ranking_2[split_val], int) and self.pred_ranking_2[split_val] == 0:
                    alpha = 1
                    weight_list = [alpha]
                    model_pred = self.borda_count([model_pred], weight_list)
                else:
                    pred_ranking_2 = (torch.sum(self.pred_ranking_2[split_val], dim=1).squeeze()).tolist()
                    alpha = 0.5
                    weight_list = [alpha, 1-alpha]
                    model_pred = self.borda_count([model_pred, pred_ranking_2], weight_list)
            else:
                alpha = 1
                weight_list = [alpha]
                model_pred = self.borda_count([model_pred], weight_list)
        
        elif self.use_pred_ranking_2:
            best_params = None
            if self.use_model_avg_rank:
                pred_ranking_2 = self.pred_ranking_2[split_val]
            else:
                if isinstance(self.pred_ranking_2[split_val], int) and self.pred_ranking_2[split_val] == 0:
                    pred_ranking_2 = [0 for _ in range(43)]
                else:
                    pred_ranking_2 = (torch.sum(self.pred_ranking_2[split_val], dim=1).squeeze()).tolist()
            alpha = 1
            weight_list = [alpha]
            model_pred = self.borda_count([pred_ranking_2], weight_list)

        # map target to prediction
        test_pred = df[df[split_col] == split_val].reset_index()
        test_pred[f"{split_val}"] = model_pred
        test_pred = test_pred[[target_type, f"{split_val}"]]
        test_pred = test_pred.set_index(target_type)

        return test_pred, best_params

    def predict_dataset_rank(self, test_model:str) -> Tuple[pd.DataFrame, dict]:
        return self._predict_target(
            split_col=self.model_name_col, 
            split_val=test_model, 
            target_type=self.dataset_col,
            pred_target=self.pred_target,
        )
        
    def predict_model_rank(self, test_dataset:set) -> Tuple[pd.DataFrame, dict]:
        df_path = f"LOVM/SWAB_inputs/{self.text_model}_{self.text_type}_target_dataset_{test_dataset}_seed_{self.seed}.csv"
        df = pd.read_csv(df_path)

        # df = df[~df['model_fullname'].str.contains('BLIP|BEIT3', regex=True)]

        # predict normalied metric
        df_list= []
        for d in np.unique(df.dataset):
            df_dataset = df.loc[df.dataset == d].copy()
            dataset_mean = df_dataset[self.pred_target].mean()
            df_dataset[f"norm_{self.pred_target}"] = df_dataset[self.pred_target] - dataset_mean
            df_list.append(df_dataset)
        df = pd.concat(df_list)

        return self._predict_target(
            df=df,
            split_col=self.dataset_col, 
            split_val=test_dataset, 
            target_type=self.model_name_col, 
            pred_target=f'norm_{self.pred_target}'
        )
    

    def predict_model_prediction(
            self, test_dataset:set
        ) -> Tuple[pd.DataFrame, dict]:

        return self._predict_target(
            split_col=self.dataset_col, 
            split_val=test_dataset, 
            target_type=self.model_name_col, 
            pred_target=self.pred_target
        )
    
    def _loo_prediction(
            self, loo_params:Iterable, prediction_func:Callable
        ) -> pd.DataFrame:

        all_test_pred = []
        for test_param in loo_params:
            test_pred, best_params = prediction_func(test_param)
            all_test_pred.append(test_pred)
        return pd.concat(all_test_pred, axis=1), best_params

    def loo_model_rank(self):

        return self._loo_prediction(self.all_datasets, self.predict_model_rank)
    


def main(): 
    model_gpt = ModelGPTPredictor(FEATURES_CSV)
    print(model_gpt.loo_dataset_rank())

if __name__ == "__main__":
    main()
