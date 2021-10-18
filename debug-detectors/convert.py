import sys
import pickle
import paddle
import torch
import numpy as np

def load_torch_model(filename):
    weights = torch.load(filename)
    state_dict = weights['state_dict']
    for k in state_dict.keys():
        state_dict[k] = state_dict[k].numpy()
    return state_dict

def convert(weights, torch_weight_name_file, paddle_weight_name_file, target_name):
    weight_name_map = {}
    with open(torch_weight_name_file) as tf:
        with open(paddle_weight_name_file) as pf:
            torch_line  =  tf.readlines()
            paddle_line  =  pf.readlines()
            for tk, pk in zip(torch_line, paddle_line):
                weight_name_map[pk.split()[0]] = tk.split()[0]

    dst = {}
    src = load_torch_model(weights)
    for k, v in weight_name_map.items():
        if ('weight' in k):
            if(src[v].ndim==4):
                dst[k] = src[v].transpose(0,1,3,2)
            elif(src[v].ndim==2):
                dst[k] = src[v].transpose(1,0)
            else:
                dst[k] = src[v]
        else:
            dst[k] = src[v]
        print("torch: ", v,"\t", src[v].shape)
        print("paddle: ", k, "\t", dst[k].shape, "\n")
    paddle.save(dst, target_name)


if __name__ == "__main__":
    weight_path = sys.argv[1]
    torch_map_name_file = sys.argv[2]
    paddle_map_name_file = sys.argv[3]
    # output name
    target_name = sys.argv[4]
    convert(weight_path, torch_map_name_file, paddle_map_name_file, target_name)
