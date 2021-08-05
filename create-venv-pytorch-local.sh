#!/bin/bash

if [ -e venv-pytorch ] ; then
    echo "Re-using existing environment"
else
    virtualenv -p /usr/bin/python3.6 venv-pytorch
    echo >> venv-pytorch/bin/activate
    echo "# Local CUDA installation" >> venv-pytorch/bin/activate
    echo "export PATH=/usr/local/cuda-11.1/bin:\${PATH}" >> venv-pytorch/bin/activate
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:\${LD_LIBRARY_PATH}" >> venv-pytorch/bin/activate
    echo >> venv-pytorch/bin/activate
    echo "# generally not needed but some samples print warnings without these:" >> venv-pytorch/bin/activate
    echo "export CUDA_INSTALL_DIR=/usr/local/cuda-11.1" >> venv-pytorch/bin/activate
    echo "export CUDNN_INSTALL_DIR=/usr/local/cuda-11.1" >> venv-pytorch/bin/activate
    # TensorRT
    # removed as MNIST sample looks for "libnvrtc.so.11.2" but we have 11.1
    #export LD_LIBRARY_PATH=/usr/local/TensorRT/lib:${LD_LIBRARY_PATH}
    #export TRT_LIB_DIR=/usr/local/TensorRT/lib
fi

source venv-pytorch/bin/activate

echo
echo "Installing PyTorch..."
echo
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

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
