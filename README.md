# PPO Research Reimplementation

## Clone project
```git clone https://github.com/Iemontine/ProximalPolicyOptimization.git```

## Installing Python 3.11.9
### On Ubuntu 22.04
```
sudo apt update
sudo apt upgrade
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
wget https://www.python.org/ftp/python/3.11.9/Python-3.11.9.tgz
tar -xf Python-3.11.9.tgz
cd Python-3.11.9
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
```
### On Windows/Mac
Download from https://www.python.org/downloads/release/python-3119/

## Virtual Environment
### Option 1: VSC builtin
* Create a VSC virtual environment with Ctrl + Shift + P -> Python: Create Environment
* Select Python 3.11.9
### Option 2: Run the following commands
* Create the virtual environment: ``python -m venv .venv``
* Activate the virtual environment
    * On Windows: ``.\myenv\Scripts\activate``
    * On Mac: ``source myenv/bin/activate``

## Install requirements
* Install requirements via requirements.txt
    * ``pip install -r requirements.txt``
    * NOTE: You may need to manually install some libraries that cause errors during installation.
