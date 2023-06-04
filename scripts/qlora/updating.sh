export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/wsl/lib/"
conda activate gptqlora

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 -U

cd AutoGPTQ
git fetch
git merge
pip install .[triton]

cd ..
cd bitsandbytes
git fetch
git merge
CUDA_VERSION=117 make cuda11x
python setup.py install

cd ..
pip install git+https://github.com/huggingface/transformers.git -U
pip install git+https://github.com/huggingface/peft.git -U
pip install git+https://github.com/huggingface/accelerate.git -U
cd gptqlora
git fetch
git merge
pip install -r requirements.txt -U
pip install protobuf==3.20.*
