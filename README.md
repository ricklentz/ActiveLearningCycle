# Active Learning with Rationales

  This project aims to answer if rationales aid in active learning.  Labeling uses resources and active learning attempts to add value by reducing the resources needed to construct new models. Thus this effort asks three questions:
  1) Does active learning with rationales reduce the labeling effort?
  2) If so, by how much (show AUC)?
  3) If so, what balance between the rationale and the document (represented by the variable c) optimizes this reduction?
  

### Instructions:
1. Run the following commands in the project's root directory to set up your data and model.

    - To run ETL pipeline that cleans data 
        `python process_step_1_read_split_save.py`
    - To run pipeline that precalculates data features 
        `python process_step_2_assign_features.py`
    - To run ML pipeline that trains classifier and saves AUC
        `python process_step_3_train_score_auc.py`

2. Run the following command in the root directory to run Jupyter-Lab for results visualization.
    `jupyter-lab`

