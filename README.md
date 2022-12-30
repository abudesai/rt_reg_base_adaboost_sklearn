Regression using Adaboost regressor in Scikit-Learn

- adaboost
- sklearn
- python
- ensemble
- feature engine
- scikit optimize
- flask
- nginx
- gunicorn
- docker

This is an Adaboost regressor for regression analysis problem. It is an example of ensemble models.

The regression model starts by fitting the regressor on the original dataset and then fits additional copies of the regressor on the same dataset.

Weights are adjusted by means of the loss parameter which is tunable.

The data preprocessing step includes:

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - MinMax scale variables prior to yeo-johnson transformation
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale data after yeo-johnson

- for target variable
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale target data after yeo-johnson

HPT based on Bayesian optimization is included for tuning Adaboost hyper-parameters.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as abalone, auto_prices, computer_activity, heart_disease, white_wine, and ailerons.

The main programming language is Python. Other tools include Tensorflow and Keras for main algorithm, feature-engine and Scikit-Learn for preprocessing, Scikit-Learn for calculating model metrics, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.
