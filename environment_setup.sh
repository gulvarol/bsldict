# Setting up manually:
conda create -n bsldict_env python=3.8

conda install pytorch=1.7.0 torchvision torchaudio cudatoolkit=10.1 -c pytorch
conda install -c conda-forge opencv
pip install scikit-learn
conda install -c conda-forge tqdm
conda install -c conda-forge matplotlib
conda install -c conda-forge youtube-dl
conda install -c anaconda wget
