# COKE: Causal Discovery with Chronological Order and Expert Knowledge in High Proportion of Missing Manufacturing Data (CIKM'24 Applied Research Track)

Welcome to the official code repository for **COKE: Causal Discovery with Chronological Order and Expert Knowledge in High Proportion of Missing Manufacturing Data (CIKM'24 Applied Research Track)**.


## Set up 
1. create conda environment for `python = 3.10.8`
2. `conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=11.8 -c pytorch -c nvidia` or see the latest version [here](https://pytorch.org/get-started/previous-versions/)
3. Install the dependency from `requirements.txt`
4. Run `python setup_CAM.py`, you need to put the file `CAM_1.0.tar.gz` in the same folder. You will need this file for install CAM. If you already install then you will see the folooing logs 
    ```
    R packages CAM and mboost have been installed
    ```

## Run Example

```
bash example.sh
```

Then you will get `causal_graph_list.csv` and `causal_graph.csv`in `reault/[time]` wich contrained the predict graph in different formmat.

**Aim Record**
If you tun on the `--record_aim` in the prompt, then you can use aim up to see the training process, the tutorial [link](https://aimstack.readthedocs.io/en/latest/quick_start/setup.html)

## Make Synthetic Data

generating process is in the folder `gendata`, you should make the data in language `R`

1. Setting the configuration of the synthetic you want to get in the file `make_data.R`  ex: number_variablese num_sampels. 

2. Successfully run the file, the defult of the configuration is:
    - number of sampels(n): 10000
    - number of variables(p): 50
    - number of machines(k): 20
  
    Then you will get
  
    ```
        ├── data
        │   ├── A.csv
        │   ├── arc.csv
        │   ├── x.csv
    ```
3. Run `rename.py` to make the start of variables(p) and machine(k) from 0

4. Run the `make_miss.py ` and setting the configuration, the default is:
    - num_recipes: 10
    - ratio_of_xfull: 0.01
    - missrate: 0.5
    - miss_range: 0.02
    - max_steps_per_recipes: 15

    Then you will get

    ```
        ├── data
        │   ├── arcs_miss(0.5)_rep(10)_xfull(100).csv
        │   ├── X_miss(0.5)_rep(10)_xfull(100).csv
    ```
    
    Make sure that the columns is start from `p{step_id}_ms{in_step_id}_m{var_id}`, such as: p0_ms0_m0, p0_ms1_m1, p0_ms2_m2,  p1_ms0_m3, p1_ms1_m4, p1_ms2_m5,...

4. Finally, run the file `imputation.py` to impute all the missing data, then you will get

    ```
        ├── data
        │   ├── X_miss(0.5)_rep(10)_xfull(100)_imp(missforest).csv
    ```
    
5. You only need following files to run the COKE

    ```
        ├── data
        │   ├── arcs_miss(0.5)_rep(10)_xfull(100).csv
        │   ├── X_miss(0.5)_rep(10)_xfull(100).csv
    ```


Data Genrating process is base from the paper "A hierarchical ensemble causal structure learning approach for wafer
manufacturing"