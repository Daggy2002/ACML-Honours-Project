conda remove -n acml --all

conda create -n acml python==3.10

conda activate acml

pip install --upgrade pip

pip install -U setuptools wheel

pip install tensorflow==2.8.4
