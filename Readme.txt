Intructions:
-------------

The .arff file contained certain formatting issues like trailing commas at the end of lines which were handled manually.
The cleaned .arff file is added in the clean_data directory.


Installation:
-------------
pip install -r requirements.txt


Directory structure:
-------------------
.
├── main.py
├── visualise_utils.py
├── approaches.py
├── requirements.txt
│
├── notebooks
    ├── EDA.ipynb
    ├── EDA.pdf
│
├── clean_data
│   ├── chronic_kidney_disease.arff
│   └── chronic_kidney_disease.info.txt
│
├── outputs
│   ├── pca_categories.png  (generated after run)
│   └── pca_numerics.png    (generated after run)


Directories Explained:
----------------------
1. clean_data - contains the clean .arff chronic_kidney_disease dataset and the metadata file
2. notebooks - contains the notebook used for performing initial analysis and the pdf of the notebook as well
3. outputs - all the visual outputs are saved here


Files Explained:
----------------
1. main.py - This file is the starting point. It contains the Feature Selection module that performs both the tasks.
2. approaches.py - This file contains the approaches that can be used to identify the key factors for a classification dataset
3. visualise_utils.py - This is a utils file containing the helper methods to visualise the factors for subtypes


Intructions to execute:
-----------------------
Please run Installation instruction before this.

1. The code can be run by the following command:
    python3 main.py -fp ./clean_data/chronic_kidney_disease.arff