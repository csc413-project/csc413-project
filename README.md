# CSC413 Project

# Environment Setup

```bash
conda create -n csc413proj python=3.11 -y
conda activate csc413proj
conda install -c conda-forge cudatoolkit=11.8 mesalib -y
cd src
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
```

If this is not enough, try the following:
```bash
sudo apt-get install libosmesa6 libgl1-mesa-glx libglfw3
````

# Running Experiments

Modify the config in `main.py` and then
```bash
python3 main.py
```
