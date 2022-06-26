#!/bin/bash

# activate rotnet anaconda environment
# anaconda
source /home/hackerman/anaconda3/etc/profile.d/conda.sh
# conda init bash
# eval "$(conda shell.bash hook)"
conda activate rotnet
echo Conda environment activated....

echo [INFO] Running RotNet main script...
python ../../main.py --dataset sonar3 --data_dir ../../../../../../datasets/sonar_wild_dataset_3/unlabeled-patches-96x96-stride-4.hdf5 --baseline_model resnet20 --batch_size 64 --epochs 200 --image_height 96 --image_width 96 --channels 1 --num_classes 4 --train --train_mode self_supervised_learning
