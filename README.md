# Active Learning with Rationales

  This project aims to answer if rationales aid in active learning.  Labeling uses resources and active learning attempts to add value by reducing the resources needed to construct models. Thus this effort asks three questions:
  1) Does active learning with rationales reduce the labeling effort against a baseline effort?
  2) If so, by how much (show AUC)?
  3) If so, what balance between the rationale and baseline document features (represented by the variable c) optimizes this reduction?
  

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

2. Run the following command in the root directory to run Jupyter Notebook for results visualization.
    `jupyter-lab visualize_results.ipynb`

    You can then use Tensorboard to visualize the different results:
    `tensorboard --logdir tensorboard_runs` and browse to http://localhost:6006/


### Python Dependiences:

The following packages are needed and can be installed using pip:
pip install pandas, random, numpy, pickle, pymagnitude, sklearn