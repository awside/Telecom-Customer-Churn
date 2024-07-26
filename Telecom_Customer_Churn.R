library(readr)
library(tidyverse)
library(dplyr)
library(skimr)
library(corrplot)
library(MASS)
library(glmnet)
library(caret)
library(Metrics)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(xgboost)
library(ggplot2)
library(factoextra)
library(cluster)
library(ggplot2)
library(pROC)

data <- read_csv("code/DATS201/w8/Telecom_Customer_Churn_Dataset-2.csv")

# Data classes and uniques
class_columns <- lapply(data, class)
unique_columns <- sapply(data, unique)

# Summaries
summary(data)
glimpse(data)
skim(data)

#----------------------------- Pre-processing ----------------------------------

# remove CustomerID since its unnecessary data
data <- data %>% dplyr::select(-CustomerID)

# Convert character columns to numeric values: 0 for 'No' and 1 for 'Yes'
data$Churn <- ifelse(data$Churn == "Yes", 1, 0)
data$HasTechSupport <- ifelse(data$HasTechSupport == "Yes", 1, 0)

# Convert categorical columns to factors with levels
data$HasTechSupport <- factor(data$HasTechSupport, levels = c(0, 1))
data$IsSeniorCitizen <- factor(data$IsSeniorCitizen, levels = c(0, 1))
data$Churn <- factor(data$Churn, levels = c(0, 1))
data$NumOfProducts <- factor(data$NumOfProducts, 
                             levels = c(1, 2, 3, 4))

#------------------------------ EDA --------------------------------------------

# Histograms

par(mfrow = c(1, 2))

hist(data$Age, main = "Age", xlab = NA)
hist(data$AnnualIncome, main = "Annual Income", xlab = NA)
hist(data$MonthlyCharges, main = "Monthly Charges", xlab = NA)
hist(data$TotalCharges, main = "Total Charges", xlab = NA)
hist(data$TenureMonths, main = "Tenure Months", xlab = NA)

# bar plots 

par(mfrow = c(1, 4))

barplot(table(data$HasTechSupport), 
        main = "Has Tech Support", 
        xlab = "", 
        ylab = "Count", 
        col = "blue")

barplot(table(data$IsSeniorCitizen), 
        main = "Is Senior Citizen", 
        xlab = "", 
        ylab = "Count", 
        col = "green")

barplot(table(data$Churn), 
        main = "Churn", 
        xlab = "", 
        ylab = "Count", 
        col = "red")

barplot(table(data$NumOfProducts), 
        main = "Number Of Products", 
        xlab = "", 
        ylab = "Count", 
        col = "orange")

# Box plots

par(mfrow = c(1, 2))

boxplot(data$Age ~ data$Churn,
        main = "Box Plot of Age by Churn",
        xlab = "Churn",
        ylab = "Age",
        col = c("red", "green"))

boxplot(data$AnnualIncome ~ data$Churn,
        main = "Box Plot of Annual Income by Churn",
        xlab = "Churn",
        ylab = "Annual Income",
        col = c("red", "green"))

boxplot(data$MonthlyCharges ~ data$Churn,
        main = "Box Plot of Monthly Charges by Churn",
        xlab = "Churn",
        ylab = "Monthly Charges",
        col = c("red", "green"))

boxplot(data$TotalCharges ~ data$Churn,
        main = "Box Plot of Total Charges by Churn",
        xlab = "Churn",
        ylab = "Total Charges",
        col = c("red", "green"))

boxplot(data$TenureMonths ~ data$Churn,
        main = "Box Plot of Tenure Months by Churn",
        xlab = "Churn",
        ylab = "Tenure Months",
        col = c("red", "green"))

# Pairs

par(mfrow = c(1, 1))

pairs(data[, c("Age", 
               "AnnualIncome", 
               "MonthlyCharges", 
               "TotalCharges", 
               "TenureMonths")], 
      main = "Pairs Plot of Numeric Columns")


#--------------------------- Data Preparation ----------------------------------

# Split the dataset into training and test sets
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(data$Churn, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

#--------------------------- Predictive Modeling -------------------------------

# Train logistic regression model
logistic_model <- glm(Churn ~ ., data = trainData, family = binomial)
summary(logistic_model)

# Predict on the test set using logistic regression
logistic_preds <- predict(logistic_model, testData, type = "response")
logistic_class <- ifelse(logistic_preds > 0.5, 1, 0)

# Logistic Regression Evaluation
logistic_class <- factor(logistic_class, levels = levels(data$Churn))
confusionMatrix(as.factor(logistic_class), testData$Churn)


# Train decision tree model
tree_model <- rpart(Churn ~ ., data = trainData, method = "class")

# Plot the decision tree
rpart.plot(tree_model)

# Predict on the test set using decision tree
tree_preds <- predict(tree_model, testData, type = "class")

# Match levels
tree_preds <- factor(tree_preds, levels = levels(testData$Churn))

# Evaluate decision tree model
confusionMatrix(tree_preds, testData$Churn)


#--------------------------- Resampling Methods --------------------------------

# Convert Churn to a factor with levels "No" and "Yes"
data$Churn <- factor(data$Churn, levels = c(0, 1), labels = c("No", "Yes"))

# Define cross-validation method
set.seed(123)  # For reproducibility
train_control <- trainControl(method = "cv", 
                              number = 10, 
                              savePredictions = TRUE, 
                              classProbs = TRUE)

# Train logistic regression model with cross-validation
logistic_model_cv <- train(Churn ~ ., 
                           data = data, 
                           method = "glm", 
                           family = "binomial", 
                           trControl = train_control)

# Print the results for logistic model using cv
print(logistic_model_cv)


# Train decision tree model with cross-validation
tree_model_cv <- train(Churn ~ ., 
                       data = data, 
                       method = "rpart", 
                       trControl = train_control)

# Print the results
print(tree_model_cv)


#------------------------- Unsupervised Learning -------------------------------

# Scaling / Standardize the data
data_scaled <- scale(data[, c("Age", 
                              "AnnualIncome", 
                              "MonthlyCharges", 
                              "TotalCharges", 
                              "TenureMonths")])
data_scaled <- as.data.frame(data_scaled)

# Summaries
skim(data_scaled)



# Perform PCA
pca <- prcomp(data_scaled, scale. = TRUE)

# Calculate the explained variance ratio
explained_variance_ratio <- pca$sdev^2 / sum(pca$sdev^2)

# Plot the explained variance ratio
explained_variance_df <- data.frame(
  PrincipalComponent = 1:length(explained_variance_ratio),
  ExplainedVariance = explained_variance_ratio
)

ggplot(explained_variance_df, aes(x = PrincipalComponent, y = ExplainedVariance)) +
  geom_line() +
  geom_point() +
  ggtitle('Explained Variance Ratio by Principal Components') +
  xlab('Principal Component') +
  ylab('Explained Variance Ratio') +
  theme_minimal()

# Calculate cumulative explained variance
cumulative_explained_variance <- cumsum(explained_variance_ratio)

# Print cumulative explained variance
print(cumulative_explained_variance)

# Get the loadings (rotation matrix)
loadings <- pca$rotation

# Print the loadings
print(loadings)

# Get the scores of the first few principal components
pca_scores <- pca$x[, 1:2]


# Determine the optimal number of clusters using the Elbow method
set.seed(123)
fviz_nbclust(pca_scores, kmeans, method = "wss")

# Apply K-means clustering
set.seed(123)
kmeans_result <- kmeans(pca_scores, centers = 3, nstart = 25)

# Visualize the clusters
fviz_cluster(kmeans_result, data = pca_scores)

# Analyze the characteristics of each customer segment
data$Cluster <- kmeans_result$cluster
cluster_summary <- data %>%
  group_by(Cluster) %>%
  summarise(across(where(is.numeric), mean, na.rm = TRUE))

print(cluster_summary)
















































