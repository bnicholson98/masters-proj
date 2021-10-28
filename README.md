**A Machine Learning Approach to Stock Market Trading**

Repo URL: https://git-teaching.cs.bham.ac.uk/mod-ug-proj-2020/bxn769

The main results can be found in the CSV files. For example, algo_dfV2-2019.csv refers to the results from the version 2 algoirthm on the 2019 data (this is the main test of the project).

If you wish to reproduce these results you will need to download the repository, this contains the full_data.csv which is a 100+MB file that is necessary for both the learning processes and the execution of the automated trading algorithm. You will also need an IDE appropriate for running Python code in.

The main scripts are simply in the 'codes' folder:
- trading_algo: Algorithm V1 on 2019 data.
- trading_algo2010: Algorithm V1 on 2010 data.
- trading_algoV2: Algorithm V2 on 2019 data.
- trading_algoV2-2010: Algorithm V2 on 2010 data.

To run any of these, you will need to
- Update the filepath at the start of the code (replace ###YOUR_PATH### with the path your downloaded repo is in). This is for calling the input data.

The other scripts in the 'codes' folder are:
- rf_model: script for performing grid search and testing random forest model.
- tech_inds_classifier: script for creating the expansive input data with all technical indicators.

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


**The following is a brief guide to the rest of the repository.**
- 'input-data' folder stores various input data forms, full_data.csv being the data used for the final algorithm.
- 'models' folder stores the machine learnt model.
- 'plots' contains a mix of plots and images used to make the report.
- 'practice-codes' contains general machine learning algorithms I made to practice the techniques I have used.
- 'refs' contains the bibliography used for the report.


**If you have any other queries or problems with the code please email me: bxn769@student.bham.ac.uk**
