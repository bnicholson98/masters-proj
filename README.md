**A Machine Learning Approach to Stock Market Trading**

This repo is split into 'development' and 'final'. While developing the project I was uploading my work to both this github and my University Gitlab, once files became too large to commit here I only pushed to the gitlab. 'development' shows all work and commits up until this point. 'final' shows the final important files and results.

The main results can be found in the CSV files. For example, algo_dfV2-2020.csv refers to the results from the version 2 algoirthm on the 2020 data.

To run any of these, you will need to
- Produce the dataset by running tech_inds_classifier.
- Create the rf model by then running rf_model.
**** Update the filepath at the start of the code (replace ###YOUR_PATH### with the path your downloaded repo is in). This is for calling the input data.

If you wish to re-run rf_model, you will need to uncomment the code labelled gridsearch (this was commented out to avoid a grid search happening at every run of the code). This process will take over 1 hour to run.
You will also need to update the path at the **start** of the code (as with the trading algo codes).
The model produced will be stored in your home directory.

If you wish to re-run tech_inds_classifier, you will need to update the filepath at the **start** of the code (as with trading algo code).

**The packages you will need to have installed for these codes are:**
- pandas
- numpy
- matplotlib
- pickle
- sklearn
- bs4
- pandas_ta
- pandas_datareader


**Additional scripts**
More scripts can be found in the 'additional' folder within 'codes'. They are:
- baselines: Where the baseline models are defined
- candlestick: Experimentary codes used to test plotting methods.
- prot_gb: Unused code for gradient boosting (not talked about in report).
- prot_rf: Initial random forest algoirthm on basic data input.
- prot_rf_no_open: Unused code of random forest with basic input data without the 'open' variable (not talked about in the report).
- rf_tech_inds: Random forest with techincal indicators for only one stock (not talked about in the report).

All these scripts, except rf_tech_inds, make use of an API key which I have removed for security. If you want to run any of these codes please email me to request for the API key.

**The additional packages needed for these codes are:**
- alpha_vantage
- glob

**If you have any other queries or problems with the code please email me: bxn769@alumni.bham.ac.uk**
