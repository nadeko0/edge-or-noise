"""tickml: research code for testing whether ML finds a tradeable
scalping edge in Bybit perpetual-futures tick data. See README.md."""
from .labeling import label_triple_barrier, label_triple_barrier_with_mae, profit_factor
from .features import add_core_features, FEATURE_COLS_FULL, FEATURE_COLS_ORDERFLOW, FEATURE_COLS_PRICEVOL
from .models import make_model
from .validation import walk_forward, walk_forward_predictions, oos_eval, permutation_test, equity_curve

__all__ = [
    "label_triple_barrier", "label_triple_barrier_with_mae", "profit_factor",
    "add_core_features", "FEATURE_COLS_FULL", "FEATURE_COLS_ORDERFLOW", "FEATURE_COLS_PRICEVOL",
    "make_model",
    "walk_forward", "walk_forward_predictions", "oos_eval", "permutation_test", "equity_curve",
]
