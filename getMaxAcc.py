import pandas as pd
import os
import argparse
import numpy as np
import torch

args = argparse.ArgumentParser()

args.add_argument("--result_dir", type=str, required=True)

args = args.parse_args()

if "ckpt/" in args.result_dir:
    args.result_dir = args.result_dir[5:]

setting_path = os.path.join("ckpt", args.result_dir, "checkpoint-best")
setting = torch.load(os.path.join(setting_path, "training_args.bin"))
print("setting")
print(setting)

result_path = os.path.join("ckpt", args.result_dir, "test")
epoch_list = os.listdir(result_path)

acc_dict = dict()
for i in epoch_list:
    with open(os.path.join(result_path,i),"r") as fp:
        acc_dict[int(i[5:-4])] = float(fp.readline().split()[-1])

reversed_dict = { y:x for x,y in acc_dict.items()}
acc_dict = sorted(acc_dict.items())

print('\n\n\tacc')
for (file, acc) in acc_dict:
    print('{}\t{}'.format(file, acc))

acc_dict = list(map(list,sorted(acc_dict)))
print("\nmax = ",max(np.array(acc_dict)[:,-1]))
print("max step = ",reversed_dict[max(np.array(acc_dict)[:,-1])])