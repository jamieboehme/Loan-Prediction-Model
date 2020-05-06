# Loan Prediction Model

This is a Python program that builds prediction models from train and test loan data sets to predict whether a loan will get approved or disapproved using logistic regression, decision tree, and random forest modeling techniques. This data and code was built and modified from existing data sets and a Python program tutorial provided by Analytics Vidhya.  

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Pre-Requisites

The following tools and libraries were used to execute this code. Given your system and versioning, you may experience bugs if there are different versions used.

- MS SQL Server hosted on AWS
     - CSV files provided should you want to modify the connection and utilize CSV's for data wrangling rather than connecting to MS SQL Server on AWS
- Python 3.7
- Python IDE: PyCharm Community Edition 2019.3
- Anaconda Interpreter 2019.03 with the following libraries
     - Pyodbc 4.0.30
     - Pandas 0.23.4
     - Numpy 1.15.4
     - Matplotlib 3.0.2
     - Seaborn 0.9.0
     - Sklearn 0.20.1

## Environment Preparation

To run this program, I recommend to first download and install PyCharm and Anaconda on your local machine. The software and documentation can be found at the following links: 
     - PyCharm: https://www.jetbrains.com/pycharm/ 
     - Anaconda: https://docs.anaconda.com/anaconda/
     
Upon download, you may find that the versions of the libraries have changed. You can alter the version of the library by following documentation to uninstall, install, and upgrade packages at https://www.jetbrains.com/help/pycharm/installing-uninstalling-and-upgrading-packages.html.

## Running Python Program

Following configuring your IDE and Interpreter, you can begin to download this repository. Before running the program, you will first need to ensure you can read from the loan train and test data sets provided.

If you have a MS SQL Server, you can insert the train_loan_data and test_loan_data data sets into your database. Documentation on how to complete this task can be found at https://docs.microsoft.com/en-us/sql/relational-databases/import-export/import-flat-file-wizard?view=sql-server-ver15.

Once this has been completed, modify the credentials in the "Connect to MS SQL Server where Data Sets are stored" section of code, and run the Python program. 

If you do not have a MS SQL Server, you can modify the connection to read from CSV with the CSV files provided. Documentation on how to read from CSV and store that data as a Pandas dataframe (as used for modeling) can be found at https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html. Ensure that the CSVs are loaded and stored in dataframes with the same name as the original Python program.

Once this has been completed, you can remove or comment out the "Connect to MS SQL Server where Data Sets are stored" section of code, and run the Python program.

## Author

Jamie Boehme
www.linkedin.com/in/jamieboehme

## References

### Code Source: Provided by Analytics Vidhya
     + Loan Prediction Dataset: https://courses.analyticsvidhya.com/courses/take/loan-prediction-practice-problem-using-python/texts/6119325-introduction-to-the-course
     
### Data Set: Provided by Analytics Vidhya
     + Train and Test Data Sets: https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/
     
### Documentation:
     + Matplotlib documentation: https://matplotlib.org/3.2.1/index.html
     + Pandas documentation: https://pandas.pydata.org/pandas-docs/stable/index.html
     + Scikit Learn documentation: https://scikit-learn.org/stable/index.html
     + Seaborn documentation: https://seaborn.pydata.org/603_ 
     + PyCharm documentation (to group code): https://www.jetbrains.com/help/pycharm/working-with-source-code.html


