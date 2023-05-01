#@title Imports
import jax

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')
from tracr.rasp import rasp

def make_length():
  all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
  return rasp.SelectorWidth(all_true_selector)

def reverse():
    length = make_length()  # `length` is not a primitive in our implementation.
    opp_index = length - rasp.indices - 1
    flip = rasp.Select(rasp.indices, opp_index, rasp.Comparison.EQ)
    reverse = rasp.Aggregate(flip, rasp.tokens)
    return reverse

def hist():
    is_same_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.EQ)
    is_same = rasp.SelectorWidth(is_same_selector)
    return is_same

def make_ones():
    return rasp.SelectorWidth(rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.EQ))

def is_match():
    return rasp.tokens=="l"

def make_frac_prevs() -> rasp.SOp:
    """Count the fraction of previous tokens where a specific condition was True.
    Eg:
        num_l("ellie")
        >> [0, 1/2, 2/3, 1/2, 2/5]
    """
    is_match_ = is_match()
    prevs = rasp.Select(rasp.indices, rasp.indices, rasp.Comparison.LEQ)
    return rasp.numerical(rasp.Aggregate(prevs, is_match_))

def make_sort_unique():
    count_less_than = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.LT)
    positions = rasp.SelectorWidth(count_less_than)
    positions_selector = rasp.Select(positions, rasp.indices, rasp.Comparison.EQ)
    return rasp.Aggregate(positions_selector, rasp.tokens)