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

# Documentation
# preprocessing.features

## FeatureGenerator
```python
FeatureGenerator(self, cols=None)
```
Generate Features for the different algorithms.

This class interface facilitates the fitting and transforming of different
datasets and gathers the numerous aggregation levels.

Parameters
----------
`cols` : `list`
       List of column names to be returned at most. Must be a subset of all
       columns that are defined within the `fit` method.

Attributes
----------
* `cols` : `list`
       Column names of the returned array.

* `dropcols` : `list`
           Column names that should be dropped, because their correlation
           with other variables is too high.

* `target` : `pd.Series`
         Series of class values/labels.

* `features` : `pd.DataFrame`
           Original features, copy of `data`.

* `item_woe` : `pd.DataFrame`
           `item_id` and associated weight of evidence.

* `color_woe` : `pd.DataFrame`
            `item_color` and associated weight of evidence.

* `brand_woe` : `pd.DataFrame`
            `brand_id` and associated weight of evidence.

* `size_woe` : `pd.DataFrame`
           `item_size` and associated weight of evidence.

* `items` : `pd.DataFrame`
        Contains general information on items. Groupby-Keys: `item_id`.

* `orders` : `pd.DataFrame`
        Contains general information on orders. Groupby-Keys: `user_id`
        `order_date`.

* `brands` : `pd.DataFrame`
        Contains general information on brands. Groupby-Keys: `brand_id`.

* `states` : `pd.DataFrame`
        Contains general information on states. Groupby-Keys: `user_state`.

* `outfeatures`: `pd.DataFrame`
             DataFrame version of the returned `numpy.ndarray`.

Methods
-------


### fit
```python
FeatureGenerator.fit(self, data, target_col)
```
Compute aggregated data according to different levels.

The different aggregation levels combine information on:
* items: group by `item_id`
* orders: group by `user_id` and `order_date`
* brands: group by `brand_id`
* states: group by `states`

Parameters
----------
`data` : `pd.DataFrame`
       Full set of information to be used. This can be the `known`
       dataset or a concatenation of `known` and `unknown`.
`target_col` : `str`
             Name of the column that contains the labels

### transform
```python
FeatureGenerator.transform(self, X, ignore_woe=True)
```
Add aggregated information to X and compute additional features.

Parameters
----------
`X` : `pd.DataFrame`
    DataFrame to be transformed.
`ignore_woe` : bool, optional
             Flag if Weight of Evidence columns should be ignored.
             Default is `True`.

Returns
-------
`out`, [y] : `np.ndarray`
           Note that `y` will only be returned if columns of `X`
           contain the target column.

### fit_transform
```python
FeatureGenerator.fit_transform(self, data, target_col, ignore_woe=True)
```
Fit data then transform it.
# preprocessing.cleaning

Clean the dataset.

Gather all functions which clean the data and compute features that do not
require aggregation.

# models

# models.tuning

Algorithms for parameter tuning.

## minimizer
```python
minimizer(objective)
```
Create a function to minimize an objective function.

This function is intended to be used as a decorator.

Parameters
----------

`objective` : `function`
              The objective function that will be optimized. This function
              should return a dictionary of the form
              `{"loss": loss, "status": hyperopt.STATUS_OK}`.
Returns
-------
`outer` : `function`
            This decorated function will be optimized over a provided
            parameter space.
