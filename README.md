# BADS 2019

## Contents
* `data`: contains `known` and `unknown` datasets.
* `eda`: module that contains scripts for **exploratory data analysis**.
* `models`: module that contains scripts that perform **parameter tuning** and generate **predictions** for different algorithms.
    + `Random Forest`
    + `Logistic Regression`
    + `Support Vector Machines`
    + `Gradient Boosted Trees`
* `predictions`: contains **predictions** of the various model runs - on github this is either empty or contains uninformative garbage :)
* `preprocessing`: module that contains scripts for **data cleaning** and **feature engineering**

## Features

![Correlation Plot of Features](https://github.com/thsis/BADS19/blob/master/eda/corrplot.png)

Some of the engineered features are still highly correlated and need to be removed before training time.

## Getting the code of this repo to runs

The first step is to install all requirements:

```bash
pip install -r requirements.txt
```

The second step is to install the repo as a package (this makes importing modules from a parent-directory possible):

```bash
pip install -e .
```
You can skip this step. However, imports will start failing. If you do not want to install random code from an untrusted repository, you can move any notebooks/scripts you want to execute into the parent directory. This should do the trick too.
