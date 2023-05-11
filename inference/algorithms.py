#@title Imports
import jax

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')
from tracr.rasp import rasp
import torch

def make_length():
  all_true_selector = rasp.Select(rasp.tokens, rasp.tokens, rasp.Comparison.TRUE)
  widths = rasp.SelectorWidth(all_true_selector)
  return widths

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

def get_decode_fn(algo: str, model_config):
    def decode_fn(logits, compiled_model):
        if algo == "reverse":
            unembed_mat = torch.eye(model_config['hidden_size'], model_config['vocab_size_out'])
            decoded = torch.argmax(torch.matmul(logits, unembed_mat), dim=1).tolist()
        else:
            decoded = []
            for i in range(logits.shape[0]):
                out=1/torch.round(logits[i,:], decimals=3)
                out=int(torch.max(torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)).item()-1)
                decoded.append(out)
        return ['bos'] + compiled_model.output_encoder.decode(decoded)[1:]
    
    return decode_fn
