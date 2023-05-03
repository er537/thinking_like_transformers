from inference.load_weights import run_pytorch_inference

TEST_INPUT = [5,4,3]

def test_reverse():
    assert run_pytorch_inference(TEST_INPUT) == list(reversed(TEST_INPUT))

if __name__=='__main__':
    test_reverse()