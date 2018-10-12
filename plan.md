# Churn predicted
business questions:
predict yes/no rides in the last 30 days
understand why

data:
build model
find features wtih strongest predictive power
define actionable steps

features to engineer:
we need column ride in last 30 day
1 = no trip in the last 30 days
user lifetime
trips per month
dollars per month

learned from eda:



# Crisp DM Workflow:

## Business understanding  
What do we want to learn?  
What problem do we want to solve?  

## Data understanding  
What data do we need?  
What data do we have?  
Frame business questions as data questions.  

# Pipeline  

## Data preparation  
Import the necessary libraries  
Read the data  
EDA  
Identify problems, missing data, wrong data, etc  
Establish process to data munging/cleaning  
Split data into training and testing sets  

## Modeling  
Preprocess and Feature Engineer  
Feature selection
Pipeline creation
Standardization  
Fit model/pipeline
Define loss function / evaluation metric
Initial evaluation metric with cross validation
Feature selection/reduction, Cross Validation
Parameter grid search (if appropriate)
 * can treat base model as paramter  
repeat as appropriate (tuning the model)  

## Evaluation
Establish relevant metrics  
Evaluate model(s) with metrics  
Feed insight back to Business Understanding  

## Deployment

### Particular Model Type Concerns:
* Linear Regression
 * Linearity
 * Plot residuals
 * Homoscedasticity, plot and het-goldfeld-quandt test
 * Normality, Q-Q plot, Jarque-Bera test
 * Multicolinearity, coefficient p-values, Variance Inflation Factor
 * Ridge / LassoHyperparameter: lambda, high-coefficient penalty
 * Higher polynomial for curve (create feature from ^x of existing)
* Logistic Regression
 * Best to model probabilities, i.e. soft classifier
 * Threshold turns soft into hard
 * Log-loss, negative log-likelihood, for evaluation
 * Log-odds modeled by linear combination of features (assumption)
* KNN
 * Euclidean, Mahatten, or Cosine distance function
 * Only hyperparameter is K, inversly proportional to variance
 * Curse of dimensionality
 * Categorial features don't work
* SVM
 * C - penalty parameter; inversely proportional to width of margin
 * Kernal type (linear, RBF, polynomial, sigmoid)
 * Degree of polynomial (if used)
 * gamma kernal kernal coefficient, 
 * Kernal trick: project to higher dimensions to get straight line, then reduce back down
 * Grid search often used


# Using Git 

Your team captain should fork the case study repository. This will be your team's "upstream repo".  

All other team members should fork the upstream repo.  

Everyone clones their own forked repo to their own local machine.  

On your local machine, create and checkout a branch to work on: git checkout -b <feature_name>. This will be your feature branch. No one works on the master branch, not even the upstream owner.  

Do your work.  

Everytime you complete an atomic piece of work: git add -p git commit -m git push origin <your feature branch>  

git add -p to interactively stage chunks of new/modified code. This is crucial to ensuring you commit only what you intend.  

git commit -m <something useful>. Your commit messages serve as documentation and communication for your team. Example of a good commit message: "add private method to feature engineering class to one-hot-encode categorical variables". Examples of a useless commit message: "stuff", "commit", "bug fix".  

git push origin feature_1, git pull -r origin master. Be explicit about which remote branch you want when you push/pull.  

Once a useful chunk of work is complete, issue a pull request to merge your branch with the upstream repo.  

The owner of the upstream repo can accept your pull request and merge it into the upstream master branch, then delete your feature branch.
Iterate frequently.  

Avoid merge conflicts by working on separable areas of code and rebasing often git pull -r origin master.  

In the end, everything will be merged to the master branch in the upstream repo. This will be your “production” code that everyone will have a copy of in the end.  

Consult the github documentation if/when you get stuck.  