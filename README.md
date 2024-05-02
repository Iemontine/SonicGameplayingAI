Running on Python 3.11.9 on Ubuntu 22.04

## Python Setup
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

* Then create a VSC virtual environment with Ctrl + Shift + P -> Python: Create Environment
* Select Python 3.11.9
* Install requirements via requirements.txt