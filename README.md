# COMP-551-Assignment-2
#### The location of all cleaned training data and code for the second assignment in COMP 551 for Jorge Diaz, Tyler Kolody, Helen You and 

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
    * Lines 74-77 and 95-99 can be swapped in order to compute train/test accuacy.
    * Lines 103-108 can be uncommented to compute a very rough indication of how well the classifier works, using 2 established predictions, each with approximately 77.5% accuracy on the test labels. 
  
