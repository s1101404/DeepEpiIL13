import os
import glob
import numpy as np
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("-in","--path_input", type=str, help="the path of input file")
parser.add_argument("-out","--path_output", type=str, help="the path of output file")
parser.add_argument("-dim","--dim", type=int, help="the dimension of input sequences")

def get_feature(input_dir, output_dir,  lenght, dim):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in tqdm(input_dir, desc="Processing", unit="file"):
        name = os.path.basename(i).split(".")[0]
        save_path = os.path.join(output_dir, name)

        data = np.loadtxt(i).reshape(1, -1, dim)
            
        if data.shape[1] < lenght:
            lenght_pad = lenght - data.shape[1]
            data_pad = np.pad(data, [(0,0), (0,lenght_pad), (0,0)], mode='constant', constant_values=0)
            np.save(save_path, data_pad)
            
        elif data.shape[1] > lenght:
            data_resize = data[:, :lenght, :]
            np.save(save_path, data_resize)
    
        else:
            np.save(save_path, data)


args = parser.parse_args()
input_folder = args.path_input
output_folder = args.path_output
dim = args.dim

lenght = 35
input_files = glob.glob(input_folder+"/*")
print(len(input_files))

get_feature(input_files, output_folder, lenght, dim)