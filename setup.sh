#!/usr/bin/env bash

# git
apt-get update && apt-get install -y git wget curl
( cd ~ && git clone git@github.com:liqi0816/6613-proj2.git && mv 6613-proj2 proj )
( cd ~/proj && git checkout liqi )

# conda
apt-get update && apt-get install -y aria2
aria2c -x 16 -s 16 --dir=/dev/shm https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
[[ -d /opt/conda ]] && command rm -r /opt/conda
bash /dev/shm/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
command rm /dev/shm/Miniconda3-latest-Linux-x86_64.sh
/opt/conda/bin/conda init
source $HOME/.bashrc

# environment
conda env create -f ~/proj/environment.yml
conda clean -y --all
conda activate clvision-challenge

# dataset
cd ~/proj

DIR="$(pwd)"
mkdir -p $DIR/cl_ext_mem
mkdir -p $DIR/submissions

apt-get update && apt-get install -y git wget unzip

echo "Downloading Core50 dataset (train/validation set)..."
MIRROR=
MIRROR="${MIRROR:-http://bias.csr.unibo.it/maltoni/download/core50}"
# wget --directory-prefix=$DIR'/core50/data/' $MIRROR/core50_128x128.zip
wget --directory-prefix=$DIR'/core50/data/' $MIRROR/core50_imgs.npz
python <<EOF
import numpy as np
npzfile = np.load('${DIR}/core50/data/core50_imgs.npz')
x = npzfile['x']
print("Writing bin for fast reloading...")
x.tofile('/dev/shm/core50_imgs.bin')
EOF
ln --symbolic /dev/shm/core50_imgs.bin $DIR'/core50/data/core50_imgs.bin'
# command rm $DIR'/core50/data/core50_imgs.npz'

echo "Downloading challenge test set..."
wget --directory-prefix=$DIR'/core50/data/' $MIRROR/core50_challenge_test.zip

echo "Unzipping data..."
# unzip $DIR/core50/data/core50_128x128.zip -d $DIR/core50/data/
unzip $DIR/core50/data/core50_challenge_test.zip -d $DIR/core50/data/
command rm $DIR/core50/data/core50_challenge_test.zip

# mv $DIR/core50/data/core50_128x128/* $DIR/core50/data/
