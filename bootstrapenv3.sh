#!/bin/bash
#mkdir /home/adivp416/tmpconda
#TMP=/home/adivp416/tmpconda
#TMPDIR=/home/adivp416/tmpconda bash /home/adivp416/public_html/Miniconda3-py39_4.10.3-Linux-x86_64.sh -u -b -p /home/adivp416/miniconda
#python -m pip install --user numpy scipy matplotlib
cd /home/adivp416/
export HOME=/home/adivp416
echo "alias pyenv=/home/adivp416/.pyenv/bin/pyenv">>/home/adivp416/.bashrc
#touch $HOME/.bashrc
#curl https://pyenv.run | bash
source $HOME/.bashrc
pyenv install 3.9.9
