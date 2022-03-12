# GCN_LSTM 
## Project details
**Interaction-aware Trajectory Prediction for Heterogeneous Traffic Agents**

**Student name**: han sun

**Supervisor**: Prof. Ko Nishino

## Documentation
The project path: /home/hsun/GCN-lstm

Corresponding singularity environment files: /home/hsun/hsun-glstm.sif

Raw dataset: /home/hsun/GCN-lstm/data/Apolloscape/prediction_test/prediction_test.txt + /home/hsun/GCN-lstm/data/Apolloscape/prediction_train/result_\*\*\*\*_\*_frame.txt

Dataset preprocess script: /home/hsun/GCN-LSTM/format_apolloscape.py

Formatted dataset: /home/hsun/GCN-lstm/data/Apolloscape/prediction_test/formatted/* + /home/hsun/GCN-lstm/data/Apolloscape/prediction_train/formatted/*

Dataset read in setting file: /home/hsun/GCN-lstm/dataset_apolloscape.py

Evaluation scripts provided by the dataset: /home/hsun/GCN-lstm/evaluation.py

Model definition: /home/hsun/GCN-LSTM/GCN_LSTM_model.py

Toolkit for GCN-LSTM：/home/hsun/GCN-LSTM/GCN_help_function.py


## How to run
**singularity shell --nv  /home/hsun/hsun-glstm.sif**

**python /home/hsun/GCN_LSTM/GCN_LSTM_main.py --[suffix name]**

## Result
Please read the paper at: /home/hsun/GCN-LSTM/GCNLSTM_Han_Sun.pdf
