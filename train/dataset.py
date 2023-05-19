import torch
import string
import random
from algorithms import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self, max_batch_size: int, input_encoder, compiled_torch_model, vocab: list = string.ascii_lowercase, max_seq_len: int = 10):
        super().__init__()
        self.batch_size = max_batch_size
        self.vocab = vocab
        self.max_seq_len = max_seq_len
        self.torch_model = compiled_torch_model
        self.input_encoder = input_encoder
    
    def __len__(self):
        return self.batch_size
        
    def __getitem__(self, idx):
        len = random.randint(2,self.max_seq_len)
        input = ['bos'] + self.get_random_input(len)
        str = self.input_encoder.encode(input)
        logits = self.torch_model(str)
        return (str, logits)

    def get_random_input(self, length):
        # choose from all lowercase letter
        letters = self.vocab
        input = [random.choice(letters) for i in range(length)]
        return input