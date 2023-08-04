# Mushroom Classification with Random Forest

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code for a simple mushroom classification using the Random Forest algorithm. The dataset used for this project is 'mushrooms.csv', which contains various features of mushrooms and their corresponding class labels.

The main goal of this project is to demonstrate how to use the Random Forest Classifier for mushroom classification and to provide a basic template that you can modify or use as a starting point for your own classification tasks.

## Requirements

Before running the code, make sure you have the following dependencies installed:

- Python 3.x
- pandas
- numpy
- scikit-learn

You can install the required libraries using pip:

```
pip install pandas numpy scikit-learn
```

## Usage

1. Clone the repository or download the 'mushrooms.csv' dataset.
2. Place 'mushrooms.csv' in the same directory as the script.
3. Run the script 'mushroom_classification.py'.

The script will load the dataset, preprocess the data by encoding categorical features using LabelEncoder, split the data into training and testing sets, train a Random Forest Classifier, and evaluate its accuracy on the test set.

## Code

```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
mushroomdata = pd.read_csv('mushrooms.csv')

# Function to encode categorical features
def encoded(features):
    encoder = LabelEncoder()
    encoder.fit(features)
    print(features.name, encoder.classes_)
    return encoder.transform(features)

# Encode all categorical columns in the dataset
for col in mushroomdata.columns:
    mushroomdata[str(col)] = encoded(mushroomdata[str(col)])

# Split data into features (x) and target (y)
y = mushroomdata['class']
x = mushroomdata.drop('class', axis=1)

# Split the data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

# Initialize and train the Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
rfc.fit(xtrain, ytrain)

# Make predictions on the test set
pred = rfc.predict(xtest)

# Calculate and print the accuracy
print(f"Accuracy: {accuracy_score(ytest, pred)*100}%")
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Feel free to use, modify, and distribute the code as per the terms of the MIT License.

## Dataset

The 'mushrooms.csv' dataset contains information about mushrooms, including features such as cap shape, cap surface, gill color, stalk shape, and others, along with the corresponding class labels (edible or poisonous).

## Contributing

If you want to contribute to this project or suggest improvements, feel free to open an issue or submit a pull request.

## Disclaimer

This project is for educational and demonstration purposes only. The accuracy of the classification model may not be suitable for production use. Use it at your own risk.
