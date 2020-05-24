# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

def run(input_size,architecture,weight_path,imnet_path,batch,workers,shared_folder_path,job_id,local_rank,global_rank,num_tasks):
    
    if Path(str(shared_folder_path)).is_dir():
        shared_folder=Path(shared_folder_path+"/evaluate/")
    else:
        raise RuntimeError("No shared folder available")
        
    train_cfg = TrainerConfig(
                    data_folder=str(data_folder_Path),
                    architecture=architecture,
                    weight_path=weight_path,
                    input_size=input_size,
                    imnet_path=imnet_path,
                    batch_per_gpu=batch,
                    workers=workers,
                    local_rank=local_rank,
                    global_rank=global_rank,
                    num_tasks=num_tasks,
                    job_id=job_id,
                    save_folder=str(shared_folder),
                )

    trainer = Trainer(train_cfg, cluster_cfg)
    

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script for FixRes models",formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-size', default=320, type=int, help='Images input size')
    parser.add_argument('--architecture', default='IGAM_Resnext101_32x48d', type=str,choices=['ResNet50', 'PNASNet' , 'IGAM_Resnext101_32x48d'], help='Neural network architecture')
    parser.add_argument('--weight-path', default='/where/are/the/weigths.pth', type=str, help='Neural network weights')
    parser.add_argument('--imnet-path', default='/the/imagenet/path', type=str, help='ImageNet dataset path')
    parser.add_argument('--shared-folder-path', default='your/shared/folder', type=str, help='Shared Folder')
    parser.add_argument('--batch', default=32, type=int, help='Batch per GPU')
    parser.add_argument('--workers', default=40, type=int, help='Numbers of CPUs')
    parser.add_argument('--job-id', default='0', type=str, help='id of the execution')
    parser.add_argument('--local-rank', default=0, type=int, help='GPU: Local rank')
    parser.add_argument('--global-rank', default=0, type=int, help='GPU: glocal rank')
    parser.add_argument('--num-tasks', default=32, type=int, help='How many GPUs are used')
    
    args = parser.parse_args()

    run(args.input_size,
        args.architecture,
        args.weight_path,
        args.imnet_path,
        args.batch,
        args.workers,
        args.shared_folder_path,
        args.job_id,
        args.local_rank,
        args.global_rank,
        args.num_tasks)
