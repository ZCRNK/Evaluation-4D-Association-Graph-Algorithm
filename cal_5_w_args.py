import os, os.path
import json
import numpy as np
import glob
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--seq_name', default="170407_haggling_a1", type=str, help='name of the current video sequence')

if __name__ == '__main__':
    args = parser.parse_args()
    seq_name = args.seq_name

    path_src = "../panoptic-toolbox/"+ seq_name +"/calibration_"+ seq_name+".json"
    src = open(path_src)
    data = json.load(src)

    path_trgt = "../results/"+ seq_name + "/calibration.json"
    trgt = open(path_trgt, "w")
    out = {}

    cams = data.get("cameras")
    num_cams = len(cams)

    cnt = 0

    for i in range(0, num_cams):

        ty = cams[i].get("type")

        if ((ty == "hd") and (cnt < 5)):

            acc = {}
            
            c = cams[i]
            
            k = c.get("K")
            k_arr = np.array(k).flatten()
            acc["K"] = k_arr.tolist()

            r = c.get("R")
            t = np.array(c.get("t")) / 100
            rt = np.concatenate((r, t), axis=1)
            rt = rt.flatten()
            acc["RT"] = rt.tolist()

            acc["imgSize"] = c.get("resolution")

            out[str(cnt)] = acc
            cnt = cnt + 1


    jsonString = json.dumps(out, indent = 4)
    trgt.write(jsonString)
    trgt.close()
