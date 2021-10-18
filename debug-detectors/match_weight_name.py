import sys
import pickle
import paddle
import torch

def load_torch_model(filename):
    weights = torch.load(filename)
    state_dict = weights['state_dict']
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].numpy()
    return state_dict

def load_paddle_model(filename):
    state_dict = paddle.load(filename)
    for k in state_dict.keys():
        state_dict[k] = state_dict[k]
    return state_dict

def match(torch_weight, paddle_weight):
    torch_state_dict = load_torch_model(torch_weight)
    paddle_state_dict = load_paddle_model(paddle_weight)
    with open('torch_state_dict.txt', 'w') as f:
        for k in torch_state_dict.keys():
            f.write(k + ' ' + str(torch_state_dict[k].shape) + '\n')
    with open('paddle_state_dict.txt', 'w') as f:
        for k in paddle_state_dict.keys():
            f.write(k + ' ' + str(paddle_state_dict[k].shape) + '\n')


if __name__ == "__main__":
    torch_weight_path = sys.argv[1]
    paddle_weight_path = sys.argv[2]
    match(torch_weight_path, paddle_weight_path)
