from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.linear_model import Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
import xgboost as xgb

FEATURES_CSV = './modelGPT/eval_table_features.csv'

G = ['intraclass_sim','inter_close','intra_close', 'superclass_metric']
C = ['text-f1', 'text-acc1']
INB = ['IN-score']
GAP_C = ['z-gap-text-f1', 'z-gap-text-acc1']
GAP_G = ['z-gap-intraclass_sim','z-gap-inter_close','z-gap-intra_close', 'superclass_metric']

ALL_FEATURES = GAP_C + GAP_G + C + G + INB 
FEATURES_SET = {"C":C, "G":G, "INB":INB, "GAP_C":GAP_C, "GAP_G":GAP_G}

DATASET_TO_REMOVE = []
MODEL_TO_REMOVE = []

MODELS = {
    'linear_regression': LinearRegression(),
    'lasso': Lasso(),
    'gbr': GradientBoostingRegressor(),
    'svr': SVR(),
    'nn': MLPRegressor(),
    'huber': HuberRegressor(epsilon=1.5, max_iter=2000, alpha=0.0001),
    'lambdamart': xgb.XGBRanker(tree_method="hist", objective="rank:pairwise", eval_metric='ndcg')
}

PRED_TARGET = 'acc1'
PRED_TARGET = 'mean_per_class_recall'
NUM_RANK = 5
MODEL_NAME_COL = 'model_fullname'
DATASET_COL = 'dataset'

PARAMETERS = {
    'linear_regression': None,
    'lasso': {
        'alpha': [0.01, 0.1, 1.0],
    }, 
    'gbr': {
        'loss': ['absolute_error', 'huber', 'quantile'],
        'alpha': [0.01, 0.1 ,1.0],
    },
    'svr': {
        'kernel': ['poly', 'rbf', 'sigmoid'],
        'degree': [2, 4, 5],
    },
    'nn': {
        'hidden_layer_sizes': [(100,), (10,), (5,)],
        'learning_rate_init': [0.001, 0.01, 0.1],
    }
}

FEATURE_ORDER_DICT = {
    'IN-score': 0,
    'z-gap-text-f1': 1,
    'z-gap-text-acc1': 2,
    'z-gap-intraclass_sim': 3,
    'z-gap-inter_close': 4,
    'z-gap-intra_close': 5,
    'text-f1': 6,
    'text-acc1': 7,
    'intraclass_sim': 8,
    'inter_close': 9,
    'intra_close': 10,
    'superclass_metric': 11
    }

FEATURE_ORDER = [
    'IN-score',
    'z-gap-text-f1',
    'z-gap-text-acc1',
    'z-gap-intraclass_sim',
    'z-gap-inter_close',
    'z-gap-intra_close',
    'text-f1',
    'text-acc1',
    'intraclass_sim',
    'inter_close',
    'intra_close',
    'superclass_metric'
    ]
