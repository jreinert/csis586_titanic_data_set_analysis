# Author: Jeremy Reinert
# Date : 2/18/2020
# Version: 1.0

"""Reads in Titanic csv file, analyzes data, and produces histograms"""

# Import Modules
import re
import pandas as pd
import matplotlib as plt
import numpy as np

# Read Titanic dataset csv file
titanic = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/carData/TitanicSurvival.csv')

# Set titanic dataframe column names
titanic.columns = ['Name', 'Survived', 'Sex', 'Age', 'Class']

# Print titanic dataframe descriptive statistics
print(titanic.describe())
print()

print('1: Show the percentage of survival among male and female passengers\n')

# Create group by sex and surivival and compute stats...
titanic_sex_survive_group = titanic.groupby(['Sex', 'Survived'])
titanic_sex_survive_group_2 = titanic.groupby(['Sex', 'Survived']).size()
# Apply lambda function to calculate percentages
titanic_sex_survive_group_percent = titanic_sex_survive_group_2.groupby(level=0).apply(lambda x: 100 * (x/x.sum()))

# Print data in tabular form
print('Raw Data: Survival Rates by Sex')
print(titanic_sex_survive_group.size())
print()
print ('Calculated Percentages: Survival Rates by Sex')
print(titanic_sex_survive_group_percent)

# Count number of male/female survivors/nonsurvivors and compute percentages
male_survive = len(titanic_sex_survive_group.get_group(('male', 'yes')))
male_nonsurvive = len(titanic_sex_survive_group.get_group(('male', 'no')))
total_male = male_survive + male_nonsurvive
male_survival_percent = (male_survive/total_male)

female_survive = len(titanic_sex_survive_group.get_group(('female', 'yes')))
female_nonsurvive = len(titanic_sex_survive_group.get_group(('female', 'no')))
total_female = female_survive + female_nonsurvive
female_survival_percent = (female_survive/total_female)

print(f'\n{male_survival_percent:.3%} of males on the Titanic survived')
print(f'{female_survival_percent:.3%} of females on the Titanic survived\n')

print('2: Show, in a histogram, the number of passengers in each class\n')
# Create a class group and compute size
class_group = titanic.groupby(['Class'])
print(class_group.size())
print('See histogram at end of output')
# Print histogram of ages by class
class_hist = titanic.Class.hist()

# Print bar chart of passenger counts by class
class_group = class_group.size().to_frame().reset_index()
class_group.columns = ['Class', 'Count']

class_bar_chart = class_group.plot.bar(x = 'Class', y = 'Count', title = 'Count of Passengers by Class', rot = 0)

print('\n3: Show the percentage of survival among each passenger class\n')
# Create group by class and survival and comput stats
class_survival_group = titanic.groupby(['Class', 'Survived'])
class_survival_group_2 = titanic.groupby(['Class', 'Survived']).size()
# Apply lambda function to calculate percentages
class_survival_group_percent = class_survival_group_2.groupby(level=0).apply(lambda x: 100 * (x/x.sum()))

# Print data in tabular form
print('Raw Data: Survival Rates by Class')
print(class_survival_group.size())
print()
print('Calculated Percentages: Survival Rates by Class')
print(class_survival_group_percent)

# Count number of survivors/nonsurvivors and compute percentages
first_class_survive = len(class_survival_group.get_group(('1st', 'yes')))
first_class_nonsurvive = len(class_survival_group.get_group(('1st', 'no')))
total_first_class = first_class_survive + first_class_nonsurvive
first_class_survival_percent = first_class_survive / total_first_class

second_class_survive = len(class_survival_group.get_group(('2nd', 'yes')))
second_class_nonsurvive = len(class_survival_group.get_group(('2nd', 'no')))
total_second_class = second_class_survive + second_class_nonsurvive
second_class_survival_percent = second_class_survive / total_second_class

third_class_survive = len(class_survival_group.get_group(('3rd', 'yes')))
third_class_nonsurvive = len(class_survival_group.get_group(('3rd', 'no')))
total_third_class = third_class_survive + third_class_nonsurvive
third_class_survival_percent = third_class_survive / total_third_class

print(f'\n{first_class_survival_percent:.3%} of passengers in 1st Class on the Titanic survived')
print(f'{second_class_survival_percent:.3%} of passengers in 2nd Class on the Titanic survived')
print(f'{third_class_survival_percent:.3%} of passengers in 3rd Class on the Titanic survived\n')

print('4: Show the percentage of survival in the following age groups: 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+\n')

# Create age bins, group by age bin and surivival, comput stats
age_bins_labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']
age_bins = [0, 9, 19, 29, 39, 49, 59, 69, 150]
age_survive = titanic.groupby([pd.cut(titanic['Age'], bins=age_bins, labels=age_bins_labels), 'Survived'])
# Apply lambda function to caclulate percentages
age_survive_2 = titanic.groupby([pd.cut(titanic['Age'], bins=age_bins, labels=age_bins_labels), 'Survived']).size()
age_survive_percent = age_survive_2.groupby(level=0).apply(lambda x: 100 * (x/x.sum()))

# Count number of survivors/nonsurvivors for each age group and compute percentages
zero_nine_survive = len(age_survive.get_group(('0-9', 'yes')))
zero_nine_nonsurvive = len(age_survive.get_group(('0-9', 'no')))
zero_nine_total = zero_nine_survive + zero_nine_nonsurvive
ten_nineteen_survive = len(age_survive.get_group(('10-19', 'yes')))
ten_nineteen_nonsurvive = len(age_survive.get_group(('10-19', 'no')))
ten_nineteen_total = ten_nineteen_survive + ten_nineteen_nonsurvive
twenty_twentynine_survive = len(age_survive.get_group(('20-29', 'yes')))
twenty_twentynine_nonsurvive = len(age_survive.get_group(('20-29', 'no')))
twenty_twentynine_total = twenty_twentynine_survive + twenty_twentynine_nonsurvive
thirty_thirtynine_survive = len(age_survive.get_group(('30-39', 'yes')))
thirty_thirtynine_nonsurvive = len(age_survive.get_group(('30-39', 'no')))
thirty_thirtynine_total = thirty_thirtynine_survive + thirty_thirtynine_nonsurvive
forty_fortynine_survive = len(age_survive.get_group(('40-49', 'yes')))
forty_fortynine_nonsurvive = len(age_survive.get_group(('40-49', 'no')))
forty_fortynine_total = forty_fortynine_survive + forty_fortynine_nonsurvive
fifty_fiftynine_survive = len(age_survive.get_group(('50-59', 'yes')))
fifty_fiftynine_nonsurvive = len(age_survive.get_group(('50-59', 'no')))
fifty_fiftynine_total = fifty_fiftynine_survive + fifty_fiftynine_nonsurvive
sixty_sixtynine_survive = len(age_survive.get_group(('60-69', 'yes')))
sixty_sixtynine_nonsurvive = len(age_survive.get_group(('60-69', 'no')))
sixty_sixtynine_total = sixty_sixtynine_survive + sixty_sixtynine_nonsurvive
seventy_plus_survive = len(age_survive.get_group(('70+', 'yes')))
seventy_plus_nonsurvive = len(age_survive.get_group(('70+', 'no')))
seventy_plus_total = seventy_plus_survive + seventy_plus_nonsurvive
total_survivors_with_age = zero_nine_total + ten_nineteen_total + twenty_twentynine_total + thirty_thirtynine_total + forty_fortynine_total + fifty_fiftynine_total + sixty_sixtynine_total + seventy_plus_total

zero_nine_survive_percent = zero_nine_survive / zero_nine_total
zero_nine_survive_percent_overall = zero_nine_survive / total_survivors_with_age
ten_nineteen_survive_percent = ten_nineteen_survive / ten_nineteen_total
ten_nineteen_survive_percent_overall = ten_nineteen_survive / total_survivors_with_age
twenty_twentynine_survive_percent = twenty_twentynine_survive / twenty_twentynine_total
twenty_twentynine_survive_percent_overall = twenty_twentynine_survive / total_survivors_with_age
thirty_thirtynine_survive_percent = thirty_thirtynine_survive / thirty_thirtynine_total
thirty_thirtynine_survive_percent_overall = thirty_thirtynine_survive/ total_survivors_with_age
forty_fortynine_survive_percent = forty_fortynine_survive / forty_fortynine_total
forty_fortynine_survive_percent_overall = forty_fortynine_survive / total_survivors_with_age
fifty_fiftynine_survive_percent = fifty_fiftynine_survive / fifty_fiftynine_total
fifty_fiftynine_survive_percent_overall = fifty_fiftynine_survive / total_survivors_with_age
sixty_sixtynine_survive_percent = sixty_sixtynine_survive / sixty_sixtynine_total
sixty_sixtynine_survive_percent_overall = sixty_sixtynine_survive / total_survivors_with_age
seventy_plus_survive_percent = seventy_plus_survive / seventy_plus_total
seventy_plus_survive_percent_overall = seventy_plus_survive / total_survivors_with_age

# Print in tabular form
print('Raw Data: Survival Rates by Age Range')
print(age_survive.size())
print()
print('Calculated Percentages: Survival Rates by Age Range')
print(age_survive_percent)
print(f'\n{zero_nine_survive_percent:.3%} of passengers age 0-9 survived')
print(f'{ten_nineteen_survive_percent:.3%} of passengers age 10-19 survived')
print(f'{twenty_twentynine_survive_percent:.3%} of passengers age 20-29 survived')
print(f'{thirty_thirtynine_survive_percent:.3%} of passengers age 30-39 survived')
print(f'{forty_fortynine_survive_percent:.3%} of passengers age 40-49 survived')
print(f'{fifty_fiftynine_survive_percent:.3%} of passengers age 50-59 survived')
print(f'{sixty_sixtynine_survive_percent:.3%} of passengers age 60-69 survived')
print(f'{seventy_plus_survive_percent:.3%} of passengers age 70+ survived')

#Raw Data Charts
age_survive_2 = age_survive_2.to_frame().reset_index()
age_survive_2.columns = ['Age Range', 'Survived', 'Count']
age_survive_3 = age_survive_2[age_survive_2.Survived == 'yes']
age_survive_4 = age_survive_2[age_survive_2.Survived == 'no']

#Percent Data Charts
age_survive_percent2 = age_survive_percent.to_frame().reset_index()
age_survive_percent2.columns = ['Age Range', 'Survived', 'Percent']
age_survive_percent3 = age_survive_percent2[age_survive_percent2.Survived == 'yes']
age_survive_percent4 = age_survive_percent2[age_survive_percent2.Survived == 'no']

print('See charts at the end of output\n')
age_survive_chart = age_survive_3.plot.bar(x = 'Age Range', title = 'Count of Survivors in Each Age Range', rot = 0)
age_survive_percent_chart = age_survive_percent3.plot.bar(x = 'Age Range', title = 'Percent of Survivors in Each Age Range', rot = 0)
age_nonsurvive_chart = age_survive_4.plot.bar(x = 'Age Range', title = 'Count of Non-Survivors in Each Age Range', rot = 0)
age_nonsurvive_percent_chart = age_survive_percent4.plot.bar(x = 'Age Range', title = 'Percent of Non-Survivors in Each Age Range', rot = 0)


print('5: Show the three common last names and list them in order of occurrence\n')

# Use regex to set pattern to capture all chars prior to the first comma 
# Group all passengers by last name, call size() to get the count, and sort the values
name_pattern = '(.+),'
name_group = titanic.groupby(titanic.Name.str.extract(name_pattern, expand = False)).size().sort_values(ascending=False)
name_group2 = name_group.to_frame().reset_index()
name_group2.columns = ['Last Name', 'Count']

# Print top 5 names in tabular form
print('Raw Data: Top 5 Most Common Last Names on the Titanic')
print(name_group2.head())

# Iterate through name_group2 dataframe to printout the three most common last names
print('\nThe three most common last names are: \n')
count = 1
most_recent_last_name_count = 0
for i, r in name_group2.iterrows():
    
    if i <= 2: 
        print(f'{r[0]} {r[1]}')
        most_recent_last_name_count = r[1]
    elif i == 3 and r[1] == most_recent_last_name_count:
            print(f'{r[0]} {r[1]}')
    if count == 5:
        break;
    count += 1

