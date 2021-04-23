//	Introduction 
Program contained in Jupyter notebook file, naive_bayes_classifer.ipynb. Please ensure that all cells are run such that all function definitions are stored in memory. 

// Functions

	preprocess(name): preprocess the given data frame and returns a dataframe containing the data and a series containing the class labels for each instance, matched on row index. 
		arguments: 
			name, String
				Accepted values of name:
					"bank": preprocesses bank.data 
					"car": preprocesses car.data 
					"carNumeric": preprocesses car.data and makes feature values numeric 
					"wdbc": preprocesses wdbc.data
					"wine": preprocesses wine.data
					"mushroom": preprocesses mushroom.data			
        
		returns: 
			X, a pandas Dataframe containing the data 
			y, a pandas series of the class labels 

	test_train(X,y,rand_state): splits the data into test and train datasets
		arguments: 
			X: data frame containg the data
			y: series containing class labels
			rand_state: a number to determine the random state of train/test split function
		
		returns:  
			X_train: A dataframe of train data  
			X_test: A dataframe of test data  
		 	y_train: a series of train classlabels 
		 	y_test: a series of test classlabels
		 	unique_vals: list of which features are numeric or categorical (if categorical the possible values are saved in a further nested list for each feature)

	train(X, y, unique_vals): trains the classifer 
		arguments:
			X: The training dataframe of the data
			y: the training series of the class labels
			unique_vals: list of which features are numeric or categorical
		
		returns:
			conditional_probs: dataframe holding the mean and standard deviation values for numeric features or the conditional probabilities for nominal features
			priors: a dictionary of the prior probabilities for each class
			class_values: a series of the unique classes and counts for each class


	predict(conditional_probs, prior, class_values, X_test, unique_vals): predicts the classes for the test data
		arguments:
			conditional_probs: A dataframe storing the mean and standard deviation values for numeric features or the conditional probabilities for nominal features
			priors: a dictionary of the prior probabilities for each class
			class_values: a series of the unique classes and counts for each class
			X_test: A dataframe of the instances we are going to predict the classes for 
			unique_vals: list of which features are numeric or categorical
		
		returns:
			predicted_class: a list of predicted classes matched on row index for each row in the X_test dataframe 

	evaluate(predicted_class, y_test): evaluates the prediction performance
		arguments:
			predicted_class: a list of predicted classes matched on row index for each row in the X_test dataframe 
			y_test: A series of the class labels of the test set - the true labels 
		returns:
			accuracy: the proportion of the correctly labeled instances


// Running the code 

	The following code is a sample run that uses the dataset bank.data.
	Note: ensure all cells of the notebook are run, including the 'import' and 'preprocess' functions cells.

	X, y = preprocess('bank')

	# split data into train and test datasets
	(X_train, X_test, y_train, y_test), unique_vals =  test_train(X, y, rand_state = 20)   

	# train data 
	conditional_probs, priors, class_values = train(X_train, y_train, unique_vals, testing=True)

	# predict on unseen data
	predicted_class = predict(conditional_probs, priors, class_values, X_test, unique_vals)

	# evaluate prediction and return accuracy
	evaluate(predicted_class, y_test)

