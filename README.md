
Check out this [Medium] post for a CRISP-DM background on this effort.

[Medium]: https://medium.com/@rlentz/active-learning-for-unstructured-data-a-crisp-dm-template-7caf8d6566a4

Please view the detailed walkthrough of the code and further discussion in the Jupyter Notebook titled `Active Learning Cycle.ipynb.`  The following provides a summary of the project motivation and results, the provided files, instructions and acknowledgments.

### Motivation of the Project, Active Learning with Rationales  

This project compares the performance of using rationales for a classification task.  Rationales in an NLP context, are words or phrases that represent the 'why' behind a position.  This project aims to answer if rationales help increase the performance of online machine learning, specifically within an Active Learning setting.  Although this project uses labeled data from the Stanford Movie Reviews dataset, the methodology should apply to any supervised text data classification task. 

### Summary of the Results  

This effort compares two versions of a logistic regression model within an Active Learning cycle.  It appears that rationales accelerate the learning process which translates into a reduced need for human annotators and less training data to achieve baseline performance.  With the Stanford Movie Review classification task, this improvement is noticeable.  However, it is still too early to make any sort of claims.  Future work should leverage several other text datasets, features, and models for a more comprehensive treatment.



### Provided Files  

  The Python files that begin with 'process_step' are used sequentially to process data, generate features using a pretrained model, perform batch training and simulated active learning, and record output for exploration using Tensorboard.

  The folder, tensorboard_runs contains AUC information for both models, over a variety of hyperparameters.  It also includes average AUC metrics where 20 randomized splits were used to get a smooth curve for easier comparison.


### Pipeline Instructions  

1. Run the following commands in the project's root directory to set up your data and model.

    - To run clean the data run 
        `python process_step_1_read_split_save.py`
    - To precalculates data features run
        `python process_step_2_assign_features.py`
    - To run train the classifier and save baseline AUC run
        `python process_step_3_baseline.py`
    - To train the classifier and save AUC run
        `python process_step_4_train_score_auc.py`

2. Run the following command in the root directory to run Jupyter Notebook.
    `jupyter-lab Active Learning Cycle.ipynb`

    You can then use Tensorboard to explore the experiment's results:
    `tensorboard --logdir tensorboard_runs` and browse to http://localhost:6006/


### Python Dependencies  
  
The following packages are needed and can be installed using pip:
`pip install pandas numpy pickle pymagnitude sklearn tensorboardx`

### Necessary Acknowledgements  

Burr Settles. 2009. “Active Learning Literature Survey” Computer Sciences Technical Report 1648. University of Wisconsin, Madison

Muhammad Yaseen of Explore Logics for his UI software engineering efforts.

Large Movie Review Dataset v1.0 @InProceedings{maas-EtAl:2011:ACL-HLT2011,
  author    = {Maas, Andrew L.  and  Daly, Raymond E.  and  Pham, Peter T.  and  Huang, Dan  and  Ng, Andrew Y.  and  Potts, Christopher},
  title     = {Learning Word Vectors for Sentiment Analysis},
  booktitle = {Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies},
  month     = {June},
  year      = {2011},
  address   = {Portland, Oregon, USA},
  publisher = {Association for Computational Linguistics},
  pages     = {142--150},
  url       = {http://www.aclweb.org/anthology/P11-1015}
}