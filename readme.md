Below you can find a outline of how to reproduce my solution for the [CIBMTR - Equity in post-HCT Survival Predictions](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions/overview) competition.  
If you run into any trouble with the setup/code or have any questions please contact me at xq_yu@sina.cn

# ARCHIVE CONTENTS
|file|usage|
|---|---|
|train.py|code to rebuild models from scratch  |
|inference.py|code to generate predictions from model binaries  |
|config.py| globle params to be set  |


# HARDWARE
```
Operating System Information:
  Platform: Ubuntu 20.04.6 LTS
  Architecture: x86_64
  CPU Cores: 12
  Total Memory: 12 GB
  GPU: NVIDIA GeForce RTX 4060 Ti
  GPU Mem:16G
```

# SOFTWARE
```
GPU:
  CUDA Version: 12.6
  Driver Version: 560.94

Python Environment:
  Python Version: 3.10.16

Python packages are detailed separately in `requirements.txt`
```

# DATA SETUP (assumes the [Kaggle API](https://github.com/Kaggle/kaggle-api) is installed)
Below are the shell commands used in each step, as run from the top level directory  
The directory below should be consistent with config.py
```sh
# Prepare data. check if your data have been put in `data_dir`(set in config.py)
mkdir data/
cd data/
kaggle competitions download -c equity-post-HCT-survival-predictions -f data_dictionary.csv
kaggle competitions download -c equity-post-HCT-survival-predictions -f train.csv
kaggle competitions download -c equity-post-HCT-survival-predictions -f test.csv

# Check `model_dir`,`data_process_dir`,`encoder_info_dir` setted in config.py are exits. If not, create a new one or replace with your trained file.
cd ../
mkdir data_preprocess 
mkdir models
mkdir encoder_info
mkdir submission  
```
# Train
The files in `model_dir`,`data_process_dir`,`encoder_info_dir` set in config.py, could be overwrite.
**==Make sure the file in these directories have been backedup if you still need.==**
``` sh
python train.py
```

# Inference
- The `model_dir`,`data_process_dir`,`encoder_info_dir` fold and related files are need for inference
- The file in `submission_dir` set in config.py could be overwrite
```sh
python inference.py
```