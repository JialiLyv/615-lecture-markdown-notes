# week1

## What is Data Science?
Learning about the world from data using computation

### Exploration
* Identifying patterns in data 
* Uses visualizations

### Inference
- Using data to draw reliable conclusions about the world 
- Uses statistics

### Prediction
- Making informed guesses about unobserved data
- Uses machine learning

# week2
### Types of ==Attributes==
### There are different types of attributes 
#### Nominal
- Examples: ID numbers, eye color, zip codes 

####  Ordinal
- Examples: rankings (e.g., taste of potato chips on a scale from 1-10), grades, height in {tall, medium, short}

#### Interval
- Examples: calendar dates, temperatures in Celsius or
Fahrenheit.

#### Ratio
- Examples: temperature in Kelvin, length, time, counts

| Attribute Type | Description | Examples | Operations
| ----------- | ----------- | ----------- | -----------
| Nominal | The values of a nominal attribute are just different names, i.e., nominal attributes provide only enough information to distinguish one object from another. (=, ) | zip codes, employee ID numbers, eye color, sex: {male, female} | mode, entropy, contingency correlation, 2 test | Title111 |
| Ordinal | The values of an ordinal attribute provide enough information to order objects. (<, >) | hardness of minerals, {good, better, best}, grades, street numbers | median, percentiles, rank cor###relation, run tests, sign tests
| Interval | For interval attributes, the differences between values are meaningful, i.e., a unit of measurement exists.(+, - ) | calendar dates, temperature in Celsius or Fahrenheit | mean, standard deviation, Pearson's correlation, t and F tests 
| Ratio | For ratio variables, both differences and ratios are meaningful. (*, /) | temperature in Kelvin, monetary quantities, counts, age, mass, length, electrical current | geometric mean, harmonic mean, percent variation 

### Types of ==data sets==
#### Record
- Data Matrix
- Document Data
- Transaction Data

#### Graph
- World Wide Web
- Molecular Structures 

#### Ordered
- Spatial Data
- Temporal Data
- Sequential Data
- Genetic Sequence Data

### Important Characteristics of ==Structured Data==
#### Dimensionality
◆ Number of attributes each object is
described with

◆ Challenge: high dimensionality (curse of dimensionality)
#### Sparsity 
◆ Only presence counts
#### #Resolution
◆ Patterns depend on the scale

### Examples of data quality problems: 
- noise and outliers
- missing values
- duplicate data

## Why Is Data ==Dirty==?
### Incomplete data may come from
-“Not applicable” data value when collected

-Different considerations between the time when the data was collected and when it is analyzed.

-Human/hardware/software problems
### Noisy data (incorrect values) may come from
-Faulty data collection instruments

-Human or computer error at data entry

-Errors in data transmission
### Inconsistent data may come from
-Different data sources

-Functional dependency violation (e.g., modify some linked data)
### Duplicate records also need data cleaning

## Measures of Location: Mean and Median

- The mean is the most common measure of the location of a set of points. 
- However, the mean is very sensitive to outliers.
- Thus, the median or a trimmed mean is also commonly used.

## Measures of Spread: Range and Variance
- Range is the difference between the max and min
- The variance or standard deviation is the most common measure of the
spread of a set of points.
- However, this is also sensitive to outliers, so that other measures are often used.

## Visualization Techniques: Histograms
### Histogram
- Usually shows the distribution of values of a single variable
- Divide the values into bins and show a bar plot of the number of objects in each bin.
- The height of each bar indicates the number of objects 
- Shape of histogram depends on the number of bins

### Two-Dimensional Histograms
- Show the joint distribution of the values of two attributes
- Example: petal width and petal length 
	- What does this tell us?

## Visualization Techniques: ==Box Plots==
- Box plots can be used to compare attributes

## Visualization Techniques: ==Scatter Plots==
### Scatter plots
- Attributes values determine the position
- Two-dimensional scatter plots most common, but can have three-dimensional scatter plots
- Often additional attributes can be displayed by using the size, shape, and color of the markers that represent the objects
- It is useful to have arrays of scatter plots can compactly
summarize the relationships of several pairs of attributes

### Visualization Techniques: ==Matrix Plots==
- Matrix plots
- Can plot the data matrix
- This can be useful when objects are ==sorted according to class==
- Typically, the attributes are ==normalized== to prevent one attribute from dominating the plot
- Plots of similarity or distance matrices can also be useful for visualizing the relationships between objects

### Visualization Techniques: ==Parallel Coordinates==
#### Parallel Coordinates
- Used to plot the attribute values of high-dimensional data
- Instead of using perpendicular axes, use a set of parallel axes
- The attribute values of each object are plotted as a point on each corresponding coordinate axis and the points are connected by a line
- Thus, each object is represented as a line
- Often, the lines representing a distinct class of objects group together, at least
for some attributes
- Ordering of attributes is important in seeing such groupings

# week3
## Distributions
### Probability Distribution
* Random quantity with various possible values
*  “Probability Distribution”:
	*  All the possible values of the quantity
	*  The probability of each of those values
*  If you can do the math, you can work out the probability distribution without ever simulating it
*  But... simulation is often easier!

###  Empirical Distribution
* “Empirical”: Based on Observations
* Observations can be from Repetitions of an Experiment
* “Empirical Distribution”
	* All observed values
	* The proportion of times each value appears

## Probability Distribution of a Statistic
* Values of a statistic vary because random samples vary
* “Sampling distribution” or “probability distribution” of the
statistic:
	* All possible values of the statistic,
	* and all the corresponding probabilities
* Can be hard to calculate
	* Either have to do the math
	* Or have to generate all possible samples and calculate
the statistic based on each sample


## Empirical Distribution of a Statistic
* Empirical distribution of the statistic:
	* Based on simulated values of the statistic
	* Consists of all the observed values of the statistic,
	* and the proportion of times each value appeared
* Good approximation to the probability distribution of the statistic
	* if the number of repetitions in the simulation is large

# week4
## Correlation Coefficient
### The Correlation Coefficient r
Measures how clustered the scatter is around a straight line

- Measures linear association 
- Based on standard units
- -1≤r≤1
	- r = 1: scatter is perfect straight line sloping up
	- r = -1: scatter is perfect straight line sloping down 
	- r = 0: No linear association; uncorrelated

Correlation r
- quadratic relation y=x2
- r=0
- Correlation measures only one kind of association - linear

### Linear Regression
A statement about x and y pairs

- Measured in standard units
- Describing the deviation of x from 0 (the average of x's)
- And the deviation of y from 0 (the average of y's)
On average, y deviates from 0 less than x deviates from 0
Not true for all points — a statement about averages

### Regression Estimate
**Goal**: Predict y using x

To find the regression estimate of y:

- Convert the given x to standard units
- Multiply by r
- That’s the regression estimate of y, but:
	- It’s in standard units
	- So convert it back to the original units of y

# week5
## Error in Estimation

- error = actual value − estimate
- Typically, some errors are positive and some negative
- To measure the rough size of the errors
	- square the errors to eliminate cancellation
	- take the mean of the squared errors
	- take the square root to fix the units
	- root mean square error (rmse)
	
## Numerical Optimization
- Numerical minimization is approximate but effective
- Lots of machine learning uses numerical minimization
- If the function mse(a, b)returns the mse of estimation
using the line “estimate = ax + b”,
	- then minimize(mse)returnsarray[a0,b0]
	- a0 is the slope and b0 the intercept of the line that minimizes the mse among lines with arbitrary slope a and arbitrary intercept b (that is, among all lines)

## Least Squares Line

- Minimizes the root mean squared error (rmse) among all lines
- Equivalently, minimizes the mean squared error (mse) among all lines
- Names:
	- “Best fit” line
	- Least squares line
	- Regression line

# week6
## Review: Residuals
- Error in regression estimate
- One residual corresponding to each point (x, y)
- residual = observed y - regression estimate of y
- In other words:
	- observed y = regression estimate + residual
	
## Residual Plot
### A scatter diagram of residuals

- Should look like an unassociated blob for linear relations
- But will show patterns for non-linear relations
- Used to check whether linear regression is appropriate
- Look for curves,  trends, changes in spread, outliers, or any other patterns

### Properties of Residuals
Residuals from a linear regression always have 
- Zero mean
	- (so rmse = SD of residuals)
- Zero correlation with x
- Zero correlation with the fitted values
- These are all true no matter what the data look like 
- Just like deviations from mean are zero on average

==_Note_: Whether or not a linear association governs the data set, the sum of the residuals (errors) for the best-fit line (or curve) is zero-always!==

## Discussion Questions
### How would we adjust our regression line...
- if the average residual were 10?
	- ==Raise the line(Shift up) by 10 units==
- if the residuals were positively correlated with x?
	- ==This means as x increases, the residual increase. Increase the slope of the line until the residuals are uncorrelated (0 correlation)with x.==
- if the residuals were above 0 in the middle and below 0 on the left and right?
	- ==Nothing==

## A Measure of Clustering
### Correlation, Revisited
- From previous lectures, “The correlation coefficient measures how clustered the points are around a straight line.”
- We can now quantify this statement.

### Variance of Fitted Values
### A Variance Decomposition
### Interpretation
- If r is close to +_1, then x has good predictive value, b/c variance of
the residuals is tiny compared to that of y
- Ifriscloseto0,thenxdoesnothavegoodpredictivevalue,b/cthe ration of variances of close to 1. May as well use uninformed prediction

### Multiple Regression
- Simple Linear Regression
	- Use one independent variable estimate the dependent variable
- Multiple Linear Regression
	- Use multiple independent variable estimate the dependent variable

### Multiple Regression
- Can include multiple variables
- Goal is to estimate one variable using several other
variables
- Used in empirical social research, market research, etc. to find out what influence different factors have on certain variable

### Assumptions
- Must be linear relationship between the dependent and independent variables
- Normally distributed error (Residual) 
- No Multicollinearity
- Homoscedasticity
	- The variance of the residuals must be constant across predicted variables

# week7

## Machine Learning
- Like human learning from past experiences.
- A computer does not have “experiences”.
- **A computer system learns from data**, which represent some “past experiences” of an application domain.
- **Goal**: learn a **target function** that can be used to predict the values of a discrete class attribute, e.g., **approve** or **not-approved**, and** high-risk** or **low risk**.
- The task is commonly called: Supervised learning, classification, or **inductive learning**.

## Supervised vs. Unsupervised Learning
### Supervised learning (classification)
- Supervision: The training data (observations, measurements, etc.) are accompanied by labels indicating the class of the observations
- New data is classified based on the training set


### Unsupervised learning (clustering)
- The class labels of training data is unknown
- Given a set of measurements, observations, etc. with the aim of establishing the existence of classes or clusters in the data

### K-Nearest Neighbours (KNN)
- KNN is a Supervised Machine Learning Algorithm
- KNN is a Nonlinear Learning Algorithm
- Instance-based learning is often termed lazy learning, as there is typically no “transformation” of training instances into more general “statements”
- Instead, the presented training data is simply stored and, when a new query instance is encountered, a set of similar, related instances is retrieved from memory and used to classify the new query instance

### Nearest neighbor method
#### Requires three things
1. The set of stored records
2. Distance Metric to compute distance between records
3. The value of k, the number of nearest neighbors to retrieve


#### To classify an unknown record:
1. Compute distance to other training records
2. Identify k nearest neighbors
3. Use class labels of nearest neighbors to determine the class label of unknown record (e.g., by taking majority vote)


### Nearest neighbor method
-  Compute distance between two points:
-  Determine the class from nearest neighbor list
-  take the majority vote of class labels among the k-nearest neighbors
-  Weigh the vote according to distance

-   k-NN classifiers are lazy learners
-   It does not build models explicitly
-   Unlike eager learners such as decision tree induction and rule-based systems
-   Classifying unknown records are relatively expensive

-  k-NN classifiers are lazy learners
-  It does not build models explicitly
-  Unlike eager learners such as decision tree induction and rule-based systems
-  Classifying unknown records are relatively expensive

## Notes on Overfitting
-  **Overfitting**: too much reliance on the training data
-  **Underfitting**: a failure to learn the relationships in the training data
-  Overfitting and underfitting cause poor **generalization** on the test set


# week8
## Prediction Problems
### Classification
- predicts categorical class labels (discrete or nominal)
- classifies data (constructs a model) based on the training set and the values (class labels) in a classifying attribute and uses it in classifying new data


### Numeric Prediction
- models continuous-valued functions, i.e., predicts unknown or missing values

## Supervised vs. Unsupervised Learning
### Supervised learning (classification)
- Supervision: The training data (observations, measurements, etc.) are accompanied by labels indicating the class of the observations

- New data is classified based on the training set

### Unsupervised learning (clustering)
- The class labels of training data is unknown
- Given a set of measurements, observations, etc. with the aim of establishing the existence of classes or clusters in the data

## Classification: Definition
### Given a collection of records (==training set== )
- Each record contains a set of ==attributes==, one of the attributes is the ==class==.

### Find a ==model== for class attribute as a function of the values of other attributes.
### Goal: _previously unseen_ records should be assigned a class as accurately as possible.
- A test set is used to determine the accuracy of the model.
- Usually, the given data set is divided into training and test sets, with training set used to build the model and test set used to validate it.

## Decision Tree Induction
### Many Algorithms:
- Hunt’s Algorithm (one of the earliest) - CART
- ID3, C4.5
- SLIQ,SPRINT

## Tree Induction
- Greedy strategy.
	- Split the records based on an attribute test that
optimizes certain criterion.
- Issues
	- Determine how to split the records
		- How to specify the attribute test condition?
		- How to determine the best split?
	- Determine when to stop splitting

### How to Specify Test Condition?
####  Depends on attribute types 
- Nominal
- Ordinal
- Continuous

####  Depends on number of ways to split
- 2-way split
- Multi-way split

### Splitting Based on Continuous Attributes
####  Different ways of handling
- ==Discretization== to form an ordinal categorical attribute
	- Static – discretize once at the beginning
	- Dynamic – ranges can be found by equal interval bucketing, equal frequency bucketing
(percentiles), or clustering.
- Binary Decision: (A < v) or (A  v)
	- consider all possible splits and finds the best cut
	- [x] can be more computationally intensive

# week9
## Model Evaluation
- Metrics for Performance Evaluation
	- How to evaluate the performance of a model?
- Methods for Performance Evaluation 、
	-  How to obtain reliable estimates?
- Methods for Model Comparison
	- How to compare the relative performance among competing models?

## Why Evaluate?
- Multiple methods are available to classify or predict
- For each method, multiple choices are available for settings
- To choose best model, need to assess each model’s performance

## Misclassification error
- Error = classifying a record as belonging to one class when it
belongs to another class.
- Error rate = percent of misclassified records out of the total records in the validation data

Metrics for Performance Evaluation

- Focus on the predictive capability of a model
- Rather than how fast it takes to classify or build models,
scalability, etc.
- ==Confusion Matrix==:

|  | PREDICTED CLASS | |  |
| ----------- | ----------- | ----------- | ----------- |
| ACTUAL CLASS | | Class=Yes | Class= No |
| | Class=Yes | a | b |
| | Class= No | c | d |

* a: TP (true positive)
* b: FN (false negative) Type 2 error 
* c: FP (false positive) Type 1 error 
* d: TN (true negative)


==Accuracy== = (a+d)/(a+b+c+d) = (TP+TN)/(TP+TN+FP+FN)

If multiple classes, 

==error rate== is:
(sum of misclassified records)/(total records)

(b+c)/(a+b+c+d)
## Limitation of Accuracy
* Consider a 2-class problem
	* Number of Class 0 examples = 9990
	* Number of Class 1 examples = 10
* If model predicts everything to be class 0, accuracy is 9990/10000 = 99.9 %
	* Accuracy is misleading because model does not detect any class 1 example
## Main Metrics

**Accuracy**: the ratio of correctly classified (TP+TN) to the total number samples

==Accuracy== = (a+d)/(a+b+c+d) = (TP+TN)/(TP+TN+FP+FN)

**Precision**: the ratio of correctly classified (TP) to the total samples **predicted** as positive samples

==Precision== = TP/(TP+FP)

**Recall** : the ratio of correctly classified (TP) divided by total number of **actual** positive samples

==Recall== = TP/(TP+FN)

**F1 score** is also known as the **F Measure**. 
The F1 score states the equilibrium between the precision and the recall.

## Methods for Performance Evaluation
- How to obtain a reliable estimate of performance?
- Performance of a model may depend on other factors besides the learning algorithm:
	- Class distribution
	- Cost of misclassification
	- Size of training and test sets

### Learning Curve
- Learning curve shows how accuracy changes with varying sample size
- Effect of small sample size: 
	- Bias in the estimate
	- Variance of estimate

### Methods of Estimation
- Holdout
	- Reserve 2/3 for training and 1/3 for testing
- Random subsampling 
	- Repeated holdout
- Cross validation
	- Partition data into k disjoint subsets
	- k-fold: train on k-1 partitions, test on the remaining one
	- Leave-one-out: k=n
- Stratified sampling
	- oversampling vs undersampling
- Bootstrap
	- Sampling with replacement

## Test of Significance
- Given two models:
	- Model M1: accuracy = 85%, tested on 30 instances
	- Model M2: accuracy = 75%, tested on 5000 instances

- Can we say M1 is better than M2?
	- How much confidence can we place on accuracy of M1 and M2?
- Can the difference in performance measure be explained as a result of random fluctuations in the test set?

## Comparing Performance of 2 Models
- Given two models, say M1 and M2, which is better?
	- M1 is tested on D1 (size=n1), found error rate = e1
	- M2 is tested on D2 (size=n2), found error rate = e2
	- Assume D1 and D2 are independent
	- If n1 and n2 are sufficiently large, then 
		- e ~ N(μ ,σ )
		- e2 ~N(μ2,σ2)
	- Approximate: σˆ = e～i(1 − e~i)/n~i

- To test if performance difference is statistically significant: d = e1 – e2
	- d~ 
	- Since D1 and D2 are independent, their variance adds up:
[Model evaluation, model selection ](https://sebastianraschka.com/blog/2018/model-evaluation-selection-part4.html)

## Decision Tree Classification
## Underfitting and Overfitting
- Two problems that can arise with models developed with Data Mining are: ==Overfitting== and Underfitting
- ==Underfitting== occurs when the model has not fully learned all the patterns in the data, resulting in poor prediction accuracy (test accuracy).
- ==Underfitting== is generally caused by the inability of the algorithm to find all patterns in the training dataset.
- In the case of a Decision Tree method the tree developed is not of sufficient depth and size to learn all the patterns present in the data

## Overfitting
- With ==Overfitting== the model’s learns the patterns in the training data very well but the model learnt cannot predict newly arriving data well
- In other words, accuracy on training dataset is high but accuracy drops drastically on newly arriving data – training set accuracy >> test set accuracy
- In the case of the Decision tree method the tree developed is too detailed (too large in size)
- ==Overfitting== is generally caused by:
	1. Noise (errors in assigning class labels) in the training
dataset
	2. Lack of sufficient data to capture certain types of patterns

### Overfitting due to Noise
Decision boundary is distorted by noise point

### Overfitting due to Insufficient Examples
Lack of data points in the lower half of the diagram makes it difficult to predict correctly the class labels of that region

- Insufficient number of training records in the region causes the decision tree to predict the test examples using other training records that are irrelevant to the classification task

## Notes on Overfitting
- Overfitting results in decision trees that are more complex than necessary
- Training error no longer provides a good estimate of how well the tree will perform on previously unseen records
- Need new ways for estimating errors

## Methods for estimating the error
- ==Re-substitution errors:== error on training (Σ e(t) )
- ==Generalization errors:== error on testing (Σ e’(t))
- Methods for estimating generalization errors:
- ==Optimistic approach:== e’(t) = e(t)
- ==Pessimistic approach:==
	- For each leaf node: e’(t) = (e(t)+0.5)
	- Total errors: e’(T) = e(T) + N × 0.5 (N: number of leaf nodes)
	- For a Tree with 30 leaf nodes and 10 errors on training (out of 1000 instances):
	Training error = 10/1000 = 1% 
	Generalization error = (10 + 30×0.5)/1000 = 2.5%
- ==Reduced error pruning (REP)==:
	- uses== validation data set== to estimate generalization error

## How to Address Overfitting...
- Post-pruning
	- Grow decision tree to its entirety
	- Trim the nodes of the decision tree in a bottom-up fashion
	- If generalization error improves after trimming, replace sub-tree by a leaf node.
	- Class label of leaf node is determined from majority class of instances in the sub-tree

## Occam’s Razor
- Given two models of similar generalization errors, one should prefer the simpler model over the more complex model
- For complex models, there is a greater chance that it was fitted accidentally by errors in data
- Therefore, one should include model complexity when evaluating a model

# week10
## Neural Networks
- Biologically inspired family of algorithms that is inspired by the human brain
- Neural Networks are used for classification, clustering and numeric prediction tasks.
- Most popular types are
- Multi Layer Perceptron (MLP) used for classification
- Radial Basis Function (RBF) used for classification and numeric prediction
- Self Organizing Map (SOM) used for clustering
- Convolutional Neural Network (CNN)used for image classification
- Long Short Term Memory (LSTM) used for modelling time series

### Artificial Neural Networks (ANN)
- Model is an assembly of inter-connected nodes and weighted links
- Output node sums up each of its input value according to the weights of its links
- Compare output node against some threshold t

### Limitations of Simple Perceptron
- Simple perceptron can be used to classify problems which are linearly separable
- For such problems a single line can be drawn which separates the two classes with zero (or near zero) error

- However simple perceptrons cannot solve non linear classification problems such as the XOR problem
- These types of problems can only be solved by adding another layer (called the hidden layer) of neurons to the network

### Solving the Logical XNOR Problem
- The XNOR problem is more difficult than the logical AND problem.
- It cannot be solved by a single neuron as it is a 2 stage process
- (X1 XNOR X2) = a1 OR a2 where a1=(X1 AND X2) and a2=(NOT X1 AND NOT X2)
- This can be seen from the following truth table

### Neural Net for Solving the Logical XNOR Problem
- a1 and a2 can be computed in parallel and so 2 neurons can be assigned to do the computation in the hidden (intermediate layer).

### General Structure of ANN

### Solving classification problems with Softmax
- Classification problems involving more than two classes are solved through the Softmax function which is implemented as an additional layer

### General Algorithm for learning ANN

- Initialize the weights (w0, w1, ..., wk)
- Compute the error at each output node (k), and the hidden node (j) connected to it.
- Now adjust the weights wjk such that wjk(new) = wjk(current)+Δwjk
	where Δwjk=rError(k)Oj
	r = learning rate parameter (0<r<1) Error(k) = the computed error at node k
	O = output of node j

### Algorithm for learning ANN
- Thus it can be seen that the observed errors are used to
adjust the weights so that the overall error is minimized
- For example if the desired output at node k is 1 and the actual output is 1.2, then the error = (1-1.2) = -0.2, so we need to decrease the weight of all incoming links starting from all nodes (e.g. j1, j2) that feed into node k
- The weight adjustment process is done iteratively until the error is below some specified threshold – this will involve scanning the data many times over

### The Loss function in Backpropagation Learning
### Backpropagation learning
- A rigorous derivation of the weight update expression using the method of
gradient descent available from: Backprop Algorithm
- Gradient descent is a commonly used for minimizing a function

### Major Parameters for Multi Layer Perceptrons
1. ==Learning rate== – this determines the size of the “steps taken” in the weight adjustment process – larger steps means learning takes place quicker but accuracy may suffer
2. ==Number of epochs== – the number of times that the training dataset is scanned – larger the value the more accurate the model (generally 100 or more)
3. The number of hidden neurons used – generally chosen as (attributes+classes)/2
4. ==Momentum== – some implementations add a term called the momentum to the current weight – this is a small fraction of the update value from the previous iteration; the momentum makes the learning process smoother

### Neural Networks - Strengths
NON-LINEARITY

- It can model non-linear systems
INPUT-OUTPUT MAPPING
- It can derive a relationship between a set of input & output responses

ADAPTIVITY

- The ability to learn allows the network to adapt to changes in the surrounding
environment

EVIDENTIAL RESPONSE

- It can provide a confidence level to a given solution
- Neural Nets work well with datasets containing noise
- Have consistently good accuracy rates across several domains
- Can be used for both supervised (classification and numeric prediction) as well as unsupervised learning

### Neural Networks - Weaknesses

- Lack the ability to explain their behaviour (unlike Decision Trees and Naïve Bayes)
- In some cases, overtraining can cause over fitting
- With large datasets training time can be large – very much larger than the Decision Tree and Naïve Bayes methods

### Overfitting with MLP

- With the Diabetes dataset: with number of neurons set to 300, Python produced 79% accuracy on the training segment and 71% accuracy on the test segment
- With number of neurons set to 100, Python produced 76% accuracy on both training and test segments
- This shows overfitting is taking place when the number of hidden neurons is high – in this case an accurate model is learn on existing data but the model cannot predict very well on new data that is arriving

### Neural Network Applications
- In general can be used for classification as well as for numeric prediction
- For classification has been used for recognizing both printed and handwritten digits
- For numeric prediction has been used for forecasting time series such as weather data (temperature, pressure, wind speed, etc), stock market prices, etc.

# week11
## What is feature selection?
Reducing the feature space by removing some of the (non-relevant) features.
Also known as:

- variable selection
- feature reduction
- attribute selection
- variable subset selection

## Feature Extraction vs Feature Selection 
- Feature Extraction
	- Generates new features based on the original dataset
		- Principal Component Analysis (PCA)
		- Vector Quantization
		- Fourier Transformation
- Feature Selection
	- Select a subset of features among the set of all features
	- Process of finding optimal set of features

## Feature Selection Approaches
- Filter Methods

Filtering approaches use a ranking or sorting algorithm to filter out those features that have less usefulness

- Wrapper Methods

The attributes subset selection is done using the learning algorithm and generally select features by directly evaluating their impact on the performance of a model.

- Embedded Methods

Embedded methods use algorithms that have built-in feature selection methods.

For instance, Lasso.

Also known as ==single factor== analysis

- The predictive power of each ==individual variable is evaluated==
	- Correlation Threshold (remove features that highly correlated with other)
	- ==Pearson’s Correlation== measures linear correlation between two variables
- ANOVA F-value (Quantitative variables) 
- Chi Square test (Categorical variables)

## Chi Square Feature Selection
Uses the chi square test whether or not a variable (the predictor feature) is independent or not of the class feature

Suppose we wish to find out whether vaccination has an effect on a particular form of pneumonia (health outcome is the class (target) variable

We compute the chi square value for each cell under the assumption of independence

## Wrapper Methods
- The predictive power of ==variable is evaluated Jointly ==
- Set of variables that performs the best
	- Subset Selection
	- Forward Selection 
	- Backward Selection
	
## Recursive Feature Extraction

- This is Python’s implementation of multivariate feature selection with a classifier as a wrapper
- Python does a greedy search through the feature space using ==backward elimination== as discussed in the lectures
- The wrapper classifier is used to guide the search by providing the chose metric (classification accuracy, area under ROC, etc.) at each point in the search.

## Forward Selection
1. In forward selection we start with ==an empty set A== of attributes
2. At each step an attribute is added and a ==performance measure== is
evaluated (for example ==Correlation map== or ==Information Gain==)
3. The attribute that produces the **best performance is added to set A**.
4. We now add each of the remaining attributes to set A and note the attribute with the highest **Information Gain (Correlation)**.
5. This attribute is now added to set A.
6. The entire process is repeated until no more attributes can be added to set A
	- i.e. at a particular round (iteration) all attributes when added decrease, rather than increase the Information Gain (Correlation).

7. The set A at the end of the process contains the set of non redundant attribute

## Backward Elimination
- Similar to forward selection but the set A initially consists of the **full set of attributes**.
- At each step we eliminate (rather than add) the attribute that leads to the **highest Information Gain**.
- The process is repeated at the iteration when every attribute that remains in **set A** leads to a **loss of Information Gain**.
- The attributes that remain in **set A** contain the list of **non redundant attributes**.

## Embedded Methods
- Built-in variable selection methods
- Regularisation –controls the value of parameters. **Less important** variable are given **lower weight** (close to zero)
	- Lasso & Ridge regression

## Principal Components Analysis (PCA)
- Finds a linear subspace (passing through the data mean) that results in the smallest (mean-square) error between the feature vectors and their projections in the data space.
- PCA de-correlates feature data via rotation.
- PCA seeks directions efficient for representation

## PCA -Summary
Advantages

- Removes Correlated Features
- Can Improves Performance and reduce Overfitting
- Reduces dimensionality leads to Improves Visualization

Disadvantages

- Independent variables become less interpretable
	- Hence, PCAs are the linear combination of your original features.
- Data standardization is must before PCA
	- PCA biased towards features with high variance, leading to false results.
