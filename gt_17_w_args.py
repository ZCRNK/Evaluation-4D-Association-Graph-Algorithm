import os, os.path
import json
import numpy as np
import glob
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--seq_name', default="170407_haggling_a1", type=str, help='name of the current video sequence')


#process one person
def process_person(kps, trgt, id):

    trgt.write(str(id)+ "\n")

    for j in range(0, 19):
        if j != 15 and j!= 17:
            trgt.write(str(kps[j*4 + 0]/100)+ "\t")
    trgt.write("\n")

    for j in range(0, 19):
        if j != 15 and j!= 17:
            trgt.write(str(kps[j*4 + 1]/100)+ "\t")
    trgt.write("\n")

    for j in range(0, 19):
        if j != 15 and j!= 17:
            trgt.write(str(kps[j*4 + 2]/100)+ "\t")
    trgt.write("\n")

    for j in range(0, 19):
        if j != 15 and j!= 17:
            trgt.write(str(kps[j*4 + 3])+ "\t")
    trgt.write("\n")


#process one .json file
def process_frame(src, trgt):

    if src.get("bodies") is not None:
        
        num_people = len(src.get("bodies"))
        trgt.write(str(num_people)+ "\n")

        for i in range(0, num_people):

            kps = src.get("bodies")[i].get("joints19")
            id_ = src.get("bodies")[i].get("id")

            process_person(kps, trgt, id_)


#MAIN
if __name__ == '__main__':
    args = parser.parse_args()
    seq_name = args.seq_name

    path_gt = "../panoptic-toolbox/"+ seq_name+"/hdPose3d_stage1_coco19"

    #iterate through files in folder and process each .json file
    num_files = len([name for name in os.listdir(path_gt)])
    print(str(num_files)+ "\n")

    path_res = "../results/"+ seq_name + "/gt.txt"

    t = open(path_res, 'w')

    t.write(str(17)+ "\t")
    t.write(str(num_files)+ "\n")

    filelist = glob.glob(os.path.join('', path_gt+'/*.json'))
    for fl in sorted(filelist, key=lambda s: s.lower()):

        path = path_gt + "/" + fl

        f = open(fl)
        data = json.load(f)

        process_frame(data, t)
        
