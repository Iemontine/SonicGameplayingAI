# PPO Research Reimplementation

## Clone project
* ```git clone https://github.com/Iemontine/ProximalPolicyOptimization.git```

<details>
<summary><h2>Installing Python 3.10.9</h2></summary>

* #### On Ubuntu 22.04
    *
        ```bash
        sudo apt update
        sudo apt upgrade
        sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev
        wget https://www.python.org/ftp/python/3.10.9/Python-3.10.9.tgz
        tar -xf Python-3.10.9.tgz
        cd Python-3.10.9
        ./configure --enable-optimizations
        make -j$(nproc)
        sudo make altinstall
        ```
* #### On Windows/Mac
    * Download from https://www.python.org/downloads/release/python-3109/</li></ul></ul>
    </details>

<details>
<summary><h2>Setting up Virtual Environment</h2></summary>

* #### Option 1: VSC builtin
    * Create a VSC virtual environment with Ctrl + Shift + P -> Python: Create Environment
    * Select Python 3.10.9
* #### Option 2: Run the following commands
    * Create the virtual environment: ```python -m venv .venv```
    * Activate the virtual environment
        * On Windows: ```.\.venv\Scripts\activate```
        * On Mac: ```source .venv/bin/activate```
</details>

<details>
<summary><h2>Install requirements</h2></summary>

* #### Install requirements via requirements.txt
    * ```pip install -r requirements.txt```
    * NOTE: You may need to manually install some libraries that cause errors during installation.
* #### Additional requirements
    ```bash
    pip install gymnasium[accept-rom-license]
    pip install stable-baselines3[extra]
    ```
    * Acquire Sonic the Hedgehog ROM. Then run: ```python3 -m retro.import ./ROM```
</details>
