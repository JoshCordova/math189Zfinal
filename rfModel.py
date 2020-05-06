import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("countyRisks.csv")

train = df.drop(range(0, 2800, 2))
y = train.deaths

feats = ['percent_fair_or_poor_health',
       '95percent_ci_low_2', '95percent_ci_high_2',
       'average_number_of_physically_unhealthy_days', '95percent_ci_low_3',
       '95percent_ci_high_3', 'average_number_of_mentally_unhealthy_days',
       '95percent_ci_low_4', '95percent_ci_high_4', 'percent_smokers',
       '95percent_ci_low_6', '95percent_ci_high_6',
       'percent_adults_with_obesity', 'percent_physically_inactive',
       'percent_excessive_drinking', '95percent_ci_low_9',
       '95percent_ci_high_9', 'num_some_college', 'population',
       'percent_some_college', 'num_associations', 'social_association_rate',
       'percent_severe_housing_problems', '95percent_ci_low_17',
       '95percent_ci_high_17', 'percent_frequent_physical_distress',
       '95percent_ci_low_24', '95percent_ci_high_24',
       'percent_frequent_mental_distress', '95percent_ci_low_25',
       '95percent_ci_high_25', 'percent_adults_with_diabetes',
       'num_food_insecure', 'percent_food_insecure',
       'percent_insufficient_sleep', '95percent_ci_low_29',
       '95percent_ci_high_29',
       'average_traffic_volume_per_meter_of_major_roadways', 'num_homeowners',
       'percent_homeowners', '95percent_ci_low_37', '95percent_ci_high_37',
       'population_2', 'percent_less_than_18_years_of_age',
       'percent_65_and_over', 'num_black', 'percent_black',
       'num_american_indian_alaska_native',
       'percent_american_indian_alaska_native', 'num_asian', 'percent_asian',
       'num_native_hawaiian_other_pacific_islander',
       'percent_native_hawaiian_other_pacific_islander', 'num_hispanic',
       'percent_hispanic', 'num_non_hispanic_white',
       'percent_non_hispanic_white', 'num_not_proficient_in_english',
       'percent_not_proficient_in_english', 'percent_female']

X = train[feats]

forest_model = RandomForestRegressor(random_state=1)

forest_model.fit(X, y)

real = df.drop(range(1, 2800, 2))

val_x = real[feats]

val_y = real['deaths']

preds = forest_model.predict(val_x)

print(mean_absolute_error(val_y, preds))
low_model = DecisionTreeRegressor(random_state=1)
low_model.fit(X, y)

preds2 = low_model.predict(val_x)

print(mean_absolute_error(val_y, preds2))