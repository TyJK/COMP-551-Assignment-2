# COMP-551-Assignment-2
#### The location of all data, weights and code for the second assignment in COMP 551 for Jorge Diaz, Tyler Kolody and Helen You

### Running each file: 

 * #### cleanData.py
    * On First Run: Will clean 'training_set.csv' and save it as 'cleaned_data.csv', requrining a 95% confidence to remove entries. 
    * The confidence required from the LangDetect library in order to remove an entry can be tweaked by changing the value of the second argument of clean_data().
    
 * #### logisticRegression.py
    * On First Run: Will use precomputed weights and the text of 'test_set_x.csv' in order to output a list of predictions and save them to a csv file. 
    * PCA can be activated by uncommenting line 59 if an additional dataset is being tested, to ensure the same number of features. 
    * New weights can be calculated by uncommenting lines 221 - 223 and commenting line 224
    * Starting at line 234, different lines can be uncommented to generate plots
 * #### neuralNetwork.py
    * On First Run: Will compute the weights of the text of 'test_set_x.csv' in order to output a list of predictions and save them to a csv file. 
    * PCA can be activated by uncommenting line 60 if an additional dataset is being tested, to ensure the same number of features. 
    * Lines 74-77 and 95-99 can be commented/uncommented in order to compute train/test accuacy.
    * Lines 103-108 can be uncommented to compute a very rough indication of how well the classifier works, using 2 established predictions, each with approximately 77.5% accuracy on the test labels. 

 * #### knn.py
    * On run it will take the file 'test_set_x.csv' as the origin from the test data, assuming it contains a header with the labels 'Id', 'Text'. This can be modified in line 140 
    * It will use as training set the data from 'cleaned_data.csv' which is created by running the file 'cleanData.py'
    * Lines 150 and 151 can be uncommented to add a validation test by spliting the training set into training and validation data.
    * Line 155 contains the parameters for the algorithm, X, Y, Test set, K, Test Indexes and name of the file where the output will be stored. 
    
 * #### adaboostTest.py
    * First run will output validation set accuracy using sklearn adaboost algorithm customized to project 2 problem and output prediction on 'test_set_x.csv',
    * save prediction result into file 'sk_adaboost.csv'.
    * It will use as training set the data from 'cleaned_data.csv' which is created by running the file 'cleanData.py'
    * To run the fully implemented adaboost algorithm, please first comment code at line 418, 419 and then uncomment the code from 423-445. 
    * Note: this algorithm does not give promising prediction result
  
