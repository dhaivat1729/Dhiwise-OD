sudo apt update
sudo apt install python3-pip python3.8-venv
python3.8 -m venv DhiwiseOD
source ./DhiwiseOD/bin/activate
python3 -m pip install -r requirements.txt
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python3 -m pip install -e .