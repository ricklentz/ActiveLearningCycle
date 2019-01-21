# Active Learning with Rationales

Please view the detailed discussion in the Jupyter Notebook titled `Active Learning Cycle.ipynb`.
  

### Instructions:
1. Run the following commands in the project's root directory to set up your data and model.

    - To run ETL pipeline that cleans data 
        `python process_step_1_read_split_save.py`
    - To run pipeline that precalculates data features 
        `python process_step_2_assign_features.py`
    - To run pipeline that trains classifier and saves AUC
        `python process_step_3_baseline.py`
    - To train the classifier and save AUC
        `python process_step_4_train_score_auc.py`

2. Run the following command in the root directory to run Jupyter Notebook.
    `jupyter-lab Active Learning Cycle.ipynb`

    You can then use Tensorboard to explore the experiment's results:
    `tensorboard --logdir tensorboard_runs` and browse to http://localhost:6006/


### Python Dependiences:

The following packages are needed and can be installed using pip:
pip install pandas, random, numpy, pickle, pymagnitude, sklearn