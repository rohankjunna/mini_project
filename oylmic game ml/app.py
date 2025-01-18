import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

# Load and clean data
teams = pd.read_csv("teams.csv")
teams = teams[["team", "country", "year", "athletes", "age", "prev_medals", "medals"]]

# Plotting relationships
sns.lmplot(x='athletes', y='medals', data=teams, fit_reg=True, ci=None)
sns.lmplot(x='age', y='medals', data=teams, fit_reg=True, ci=None)
teams.plot.hist(y="medals")

# Handling missing values
print(teams[teams.isnull().any(axis=1)].head(20))
teams = teams.dropna()

# Splitting data into train and test sets
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

print("Train shape:", train.shape)  # About 80% of the data
print("Test shape:", test.shape)    # About 20% of the data

# Training the model
reg = LinearRegression()
predictors = ["athletes", "prev_medals"]
reg.fit(train[predictors], train["medals"])

# Making predictions
predictions = reg.predict(test[predictors])
test["predictions"] = predictions

# Post-processing predictions
test.loc[test["predictions"] < 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()

# Calculating error
error = mean_absolute_error(test["medals"], test["predictions"])
print("Mean Absolute Error:", error)

# Analyzing errors
print(teams.describe()["medals"])
print(test[test["team"] == "USA"])
print(test[test["team"] == "FRA"])

errors = (test["medals"] - predictions).abs()
error_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()
error_ratio = error_by_team / medals_by_team
error_ratio = error_ratio[np.isfinite(error_ratio)]

# Plotting error ratio distribution
error_ratio.plot.hist()
plt.show()

