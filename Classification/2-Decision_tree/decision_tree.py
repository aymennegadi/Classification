import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#now we import the Titanic dataset
df = pd.read_csv("Titanic-Dataset.csv")

#General information about the dataset
print("First 5 rows (head by default provide 5 first)")
print(df.head())
print("\nColumn names:")
print(df.columns)
print("\nDataset Information:")
print(df.info())
print("\nDataset Shape:")
print(df.shape)
print("\nDataset Shape:")
print(df.describe())
print("\nMissing Values:")
print(df.isnull().sum())

#clean the data set:
#so we remove the cabin column
#(axis = 0  → Rows)
#(axis = 1  → Columns)
df.drop("Cabin", axis=1, inplace=True)

#replacing the age missing values with the median
#fillna() replaces every missing value,and returns a new column.
df["Age"] = df["Age"].fillna(df["Age"].median())

#Replace missing Embarked values with the most frequent value
##mode() returns a list, To extract the actual value "S", we use [0]
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


print("\nMissing Values After Cleaning:")
#isnull()=> Checks every cell
#So sum() counts how many True values there are.
print(df.isnull().sum())

# Remove unnecessary columns
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

print("\nDataset after removing unnecessary columns:")
print(df.head())

#These are strings (text).
#A Decision Tree can only work with numerical values.
#we must convert these text values into numbers.
#This process is called 'Encoding'.

#convert the sex column into numbers
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})
print("\nAfter encoding sex")
print(df.head())

## Convert the Embarked column into numbers
df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})
print("\nAfter encoding Embarked:")
print(df.head())

# Separate the features and the target
X = df.drop("Survived", axis=1)
y = df["Survived"]

print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())



# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2, #20% testing data
    random_state=42
)
print("Training set:", X_train.shape)
print("Testing set:", X_test.shape)

#create the decision tree model
model = DecisionTreeClassifier(random_state=42)

#now train the model
model.fit(X_train,y_train)
print("\nDecision Tree model trained successfully!")


# Predict the survival of the testing passengers
y_pred = model.predict(X_test)

print("\nFirst 10 predictions:")
print(y_pred[:10])

print("\nFirst 10 actual values:")
print(y_test[:10].values)



# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(accuracy)



# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


#Classification Repor
print("\nClassification Report:")
print(classification_report(y_test, y_pred))