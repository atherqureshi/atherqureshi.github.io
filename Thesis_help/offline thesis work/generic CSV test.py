
#importing practice datasets from Sklearn! 
#values are in csv
#(values, feature Names, TargetEnumerator, TargetEnumeratorNames)
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer

#we want csv versions of these datasets to easily visualize in d3
import numpy
import pandas

#machine learning framework from python
from sklearn.tree import DecisionTreeClassifier

#tools to create the input for d3 (JSON versions of DecisionTree)
import json
import io

# this function creates json version of decisionTreeClassifier, that we able to 
# visualize in d3 (FOUND ONLINE, and I made some slight modifications)
def rules(clf, features, labels, node_index=0):
    """Structure of rules in a fit decision tree classifier

    Parameters
    ----------
    clf : DecisionTreeClassifier
        A tree that has already been fit.

    features, labels : lists of str
        The names of the features and labels, respectively.

    """
    node = {}
    if clf.tree_.children_left[node_index] == -1:  # indicates leaf
        count_labels = zip(clf.tree_.value[node_index, 0], labels)
        node['name'] = ', '.join(('{} of {}'.format(int(count), label)
                                  for count, label in count_labels))
        node['type'] = "black"
    else: #non leaf / so parent
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        #round values to 2 decimal places
        threshold = str(round(threshold,2))
        node['name'] = '{} > {}'.format(feature, threshold)
        node['type'] = "black"
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node


def generateJSON(data):
	#create decisionTree object
    clf = DecisionTreeClassifier(max_depth=5)
    #run fit, which creates the decision tree based the data
    clf.fit(data.data, data.target)
    #run rules function to get JSON version of Decision tree from sklearn 
    JSONString = rules(clf, data.feature_names, data.target_names)
    #write JSON File with appropriate name to disk
    json.dump(JSONString, open(data.name +'_tree.json', 'wb'))
    #IU Purposes
    print("generated " + data.name)
    return

#returns an object that can be used to create decision tree
#data.feature_names = list of feature/column names
#data.target_names = list of classifications, somewhat like enumeration
#data.data is a 2D array of NUMBERS [x][y] x is the row in csv, and y is the column
#data.target is a 1D array of NUMBERS that is a simple number classifcation for each row [0 1 1 0]
#Format for CSV is: 1st Column is ID for row, 2nd Column is Target values, 3rd column on is features
def generateDictionaryFromCSV(csvFilePath):
	input_file = csvFilePath
	dataFile = pandas.read_csv(input_file, header = 0)
	feature_names = list(dataFile.columns.values)
	dataFile = dataFile._get_numeric_data()
    return
    #numpy_array = dataFile.as_matrix()

generateDictionaryFromCSV('breast_cancer.csv')


#load the data sets
Iris_data = load_iris()
BreastCancer_data = load_breast_cancer()

#add name key to dictionarys to describe dataset for readability in the JSON when we try to output in d3
Iris_data['name'] = 'Iris'
BreastCancer_data['name'] = 'Breast_Cancer'

#Decision Trees
generateJSON(Iris_data) #Decision Tree is fine (Decision Tree since descrete Set of Values as target)
generateJSON(BreastCancer_data)



    


