# Alphabet Soup Charity Neural Network Challenge

## Background:
The nonprofit foundation Alphabet Soup aims to enhance its funding allocation process by predicting the success of organizations that receive funding. In collaboration with Alphabet Soup's business team, we are tasked with creating a binary classifier using machine learning and neural networks. This model will predict whether an organization funded by Alphabet Soup will be successful based on various features in the dataset.

## Dependencies:
    pip install pandas
    pip install scikit-learn
    pip install tensorFlow
    pip install keras

## Project Structure:
### Step 3: Preprocess the Data:
  + Read in the charity_data.csv to a Pandas DataFrame.
  + Identify target and feature variables.
  + Drop unnecessary columns (EIN and NAME).
  + Bin rare categorical variables into a new value, 'Other.'
  + Encode categorical variables using pd.get_dummies().
  + Split data into features array (X) and target array (y).
  + Scale training and testing features using StandardScaler.

### Step 2: Compile, Train, and Evaluate the Model:
  + Create a neural network model using TensorFlow and Keras.
  + Design hidden layers with appropriate activation functions.
  + Compile and train the model, saving weights every five epochs.
  + Evaluate the model on test data to calculate loss and accuracy.
  + Save results to an HDF5 file named AlphabetSoupCharity.h5.

### Step 3: Optimize the Model:
Experiment with various optimization strategies:
  + Adjust input data by modifying columns, creating more bins, etc.
  + Modify neural network architecture by adding neurons, layers, or changing activation functions.
  + Adjust training parameters such as epochs.
  + Create a new Colab file named AlphabetSoupCharity_Optimization.ipynb and design an optimized neural network model.
  + Save optimized results to AlphabetSoupCharity_Optimization.h5.

## Conclusions:
The initial model yielded promising results with a loss of 0.5562 and an accuracy of 73.11%. The training process included 268 epochs with an average duration of 462ms per epoch and 2ms per step.

In contrast, the optimized model results showed a loss of 0.5549 and an accuracy of 72.78%. Despite implementing changes such as adding hidden layers and nodes, as well as adjusting the optimizer, the optimized model performed slightly worse. The accuracy did not surpass 72%, indicating that the optimization efforts did not achieve the target accuracy of higher than 75%.

These findings suggest that further exploration and experimentation may be needed to identify the most effective configurations for improving model performance.

## Results:
+ Initial Model Accuracy: 73.11%

  ![image](https://github.com/ashley-ley/deep-learning-challenge/assets/132225987/17f78419-1338-43ca-9140-d6810ccf5fcc)
+ Optimized Model Accuracy: 72.78%

  ![image](https://github.com/ashley-ley/deep-learning-challenge/assets/132225987/a9a1b9eb-efec-48a1-89dc-e827e4364799)

## Contributions
Feel free to contribute to this project by forking the repository and submitting pull requests. If you have questions, reach out through the GitHub repository.
