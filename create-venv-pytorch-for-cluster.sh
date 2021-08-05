#!/bin/bash

if [ -e venv-pytorch ] ; then
    echo "Re-using existing environment"
else
    virtualenv -p /usr/bin/python3.7 venv-pytorch
    echo >> venv-pytorch/bin/activate
    echo "# Local CUDA installation" >> venv-pytorch/bin/activate
    echo "export PATH=/spinning/jwagner/cuda-10.2/bin:\${PATH}" >> venv-pytorch/bin/activate
    echo "export LD_LIBRARY_PATH=/spinning/jwagner/cuda-10.2/lib64:\${LD_LIBRARY_PATH}" >> venv-pytorch/bin/activate
    echo >> venv-pytorch/bin/activate
    echo "# generally not needed but some samples print warnings without these:" >> venv-pytorch/bin/activate
    echo "export CUDA_INSTALL_DIR=/spinning/jwagner/cuda-10.2" >> venv-pytorch/bin/activate
    echo "export CUDNN_INSTALL_DIR=/spinning/jwagner/cuda-10.2" >> venv-pytorch/bin/activate
fi

source venv-pytorch/bin/activate

echo
echo "Installing PyTorch..."
echo
pip install torch torchvision torchaudio

for I in \
jupyter \
numpy \
scipy \
matplotlib \
sklearn \
pandas \
ipython \
cython \
matplotlib \
numpy \
dyNET \
h5py \
transformers \
pytorch-lightning \
pytorch-nlp \
tensorboard \
lightning-transformers \
; do
    echo
    echo "Installing $I"
    echo
    pip install $I
done

echo "Please activate environment and install NVIDIA apex from github"
