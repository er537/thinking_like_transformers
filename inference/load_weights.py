#@title Imports
import jax
import numpy as np

# The default of float16 can lead to discrepancies between outputs of
# the compiled model and the RASP program.
jax.config.update('jax_default_matmul_precision', 'float32')

from tracr.compiler import compiling
import torch

from model.transformer_model import TransformerModel
from inference.algorithms import *


def extract_config(model, act_fn = "relu"):
    model_config = {'activation_function': act_fn}
    for key,val in model.model_config.__dict__.items():
        if key == 'activation_function':
            continue
        model_config[key]=val
    
    model_config["max_seq_len"] = model.params["pos_embed"]['embeddings'].shape[0] 
    model_config["vocab_size"] = model.params["token_embed"]['embeddings'].shape[0] # Vocab size plus 2 for BOS and PAD
    model_config["vocab_size_out"] = model_config["vocab_size"] - 2
    model_config["hidden_size"] = model.params["token_embed"]['embeddings'].shape[1]
    return model_config 

def build_state_dict(model):
    sd = {}
    for name in model.params:
        if 'transformer' in name:
            _, layer, module, param = name.split('/')
            layer_num = layer.split('_')[1]
            sd[f"layers.{layer_num}.{module}.{param}.weight"] = torch.transpose(torch.tensor(np.array(model.params[name]['w'])), 0, 1)
            sd[f"layers.{layer_num}.{module}.{param}.bias"] = torch.tensor(np.array(model.params[name]['b']))
        else:
            sd[f"{name}.embeddings"] = torch.tensor(np.array(model.params[name]['embeddings']))
    
    return sd


def run_pytorch_inference(input: list, algo_name: str='reverse'):
    bos = "bos"
    compiled_model = compiling.compile_rasp_to_model(
        eval(f"{algo_name}()"),
        vocab={0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
        max_seq_len=9,
        compiler_bos=bos,
    )
    model_config = extract_config(compiled_model)
    state_dict = build_state_dict(compiled_model)
    torch_model = TransformerModel(model_config).eval()
    torch_model.load_state_dict(state_dict)
    
    x = compiled_model.input_encoder.encode(["bos"] + input)
    out = torch_model(x)
    # decode is algorithm specific
    decode_fn = get_decode_fn(algo_name, model_config)
    decoded_output = decode_fn(out, compiled_model)

    return decoded_output

if __name__=="__main__":
    run_pytorch_inference([5,4,3])
