# Multiple Linear Regression

![](https://miro.medium.com/max/1238/1*r3aOsJoXHX7uC2nxn2lygQ.png)

 **Linear regression involving multiple variables is called “multiple linear regression” or multivariate linear regression.** The steps to perform multiple linear regression are almost similar to that of simple linear regression. The difference lies in the evaluation. You can use it to find out which factor has the highest impact on the predicted output and how different variables relate to each other.
 
 # Steps Involved in any Multiple Linear Regression Model
 
-  **Step #1:** Data Pre Processing
    -  Importing The Libraries.
    -  Importing the Data Set.
    -  Encoding the Categorical Data.
    -  Avoiding the Dummy Variable Trap.
    -  Splitting the Data set into Training Set and Test Set.
- **Step #2:** Fitting Multiple Linear Regression to the Training set
- **Step #3:** Predicting the Test set results.

# Assumption of Regression Model :

- **Linearity:** The relationship between dependent and independent variables should be linear.
- **Homoscedasticity:** Constant variance of the errors should be maintained.
- **Multivariate normality:** Multiple Regression assumes that the residuals are normally distributed.
- **Lack of Multicollinearity:** It is assumed that there is little or no multicollinearity in the data.

### Dummy Variable –
As we know in the Multiple Regression Model we use a lot of categorical data. Using Categorical Data is a good method to include non-numeric data into respective Regression Model. Categorical Data refers to data values which represent categories-data values with the fixed and unordered number of values, for instance, gender(male/female). In the regression model, these values can be represented by Dummy Variables.

These variable consist of values such as 0 or 1 representing the presence and absence of categorical value.

![](https://media.geeksforgeeks.org/wp-content/uploads/reg1-1.png)

### Dummy Variable Trap –
The Dummy Variable Trap is a condition in which two or more are Highly Correlated. In the simple term, we can say that one variable can be predicted from the prediction of the other. The solution of the Dummy Variable Trap is to drop one the categorical variable. So if there are m Dummy variables then m-1 variables are used in the model.

```
D2 = D1-1   
 Here D2, D1 = Dummy Variables
 ```
 
### Method of Building Models :

- All-in
- Backward-Elimination
- Forward Selection
- Bidirectional Elimination
- Score Comparison

#### Backward-Elimination :

- **Step #1 :** Select a significant level to start in the model.
- **Step #2 :** Fit the full model with all possible predictor.
- **Step #3 :** Consider the predictor with highest P-value. If P > SL go to STEP 4, otherwise model is Ready.
- **Step #4 :** Remove the predictor.
- **Step #5 :** Fit the model without this variable.

#### Forward-Selection :

- **Step #1 :** Select a significance level to enter the model(e.g. SL = 0.05)
- **Step #2 :** Fit all simple regression models y~ x(n). Select the one with the lowest P-value .
- **Step #3 :** Keep this variable and fit all possible models with one extra predictor added to the one(s) you already have.
- **Step #4 :** Consider the predictor with lowest P-value. If P < SL, go to Step #3, otherwise model is Ready.
