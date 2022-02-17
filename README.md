 Summary of our Data

Before we get into the deep visualizations, we want to make sure how our data looks like right? This will better help us have a better grasp as to how we should work with our data later throughout the project.

Questions we could Ask Ourselves:

    Columns and Observations: How many columns and observations is there in our dataset?
    Missing data: Are there any missing data in our dataset?
    Data Type: The different datatypes we are dealing in this dataset.
    Distribution of our Data: Is it right-skewed, left-skewed or symmetric? This might be useful especially if we are implementing any type of statistical analysis or even for modelling.
    Structure of our Data: Some datasets are a bit complex to work with however, the tidyverse package is really useful to deal with complex datasets.
    Meaning of our Data: What does our data mean? Most features in this dataset are ordinal variables which are similar to categorical variables however, ordering of those variables matter. A lot of the variables in this dataset have a range from 1-4 or 1-5, The lower the ordinal variable, the worse it is in this case. For instance, Job Satisfaction 1 = "Low" while 4 = "Very High".
    Label: What is our label in the dataset or in otherwords the output?


Summary:

    Dataset Structure: 1470 observations (rows), 35 features (variables)
    Missing Data: Luckily for us, there is no missing data! this will make it easier to work with the dataset.
    Data Type: We only have two datatypes in this dataset: factors and integers
    Label" Attrition is the label in our dataset and we would like to find out why employees are leaving the organization!
    Imbalanced dataset: 1237 (84% of cases) employees did not leave the organization while 237 (16% of cases) did leave the organization making our dataset to be considered imbalanced since more people stay in the organization than they actually leave.
