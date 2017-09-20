    
#importing practice datasets from Sklearn! 
#values are in csv
#(values, feature Names, TargetEnumerator, TargetEnumeratorNames)
from sklearn.datasets import load_iris
from sklearn.datasets import load_breast_cancer
from numpy import array

#we want csv versions of these datasets to easily visualize in d3
import numpy
import pandas

#to create CSV from JSON
import csv
import json

#machine learning framework from python
from sklearn.tree import DecisionTreeClassifier

#tools to create the input for d3 (JSON versions of DecisionTree)
import json
import io

#tools to edit HTML files
from bs4 import BeautifulSoup

#copy a file
from shutil import copyfile
import os

#command line argument 
import sys

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


#returns the FileName of the tree
#data.feature_names = list of feature/column names
#data.target_names = list of classifications, somewhat like enumeration
#data.data is a 2D array of NUMBERS [x][y] x is the row in csv, and y is the column
#data.target is a 1D array of NUMBERS that is a simple number classifcation for each row [0 1 1 0]
#Format for CSV is: 1st Column is ID for row, 2nd Column is Target values, 3rd column on is features
def createJSONTreefromData(csvFilePath):
	input_file = csvFilePath
	dataFile = pandas.read_csv(input_file, header = 0, index_col=False, na_filter=False)
    #feature_names is now contains a list of all column values
	feature_names = list(dataFile.columns.values)
        #slice the last NAN item from list
        #feature_names = feature_names[:-1]
        target_column = feature_names[1]

        #remove first 3 columns
        feature_names.pop(0)
        feature_names.pop(0)
        feature_names.pop(0)

    #The 2nd column in the CSV is the target classifciations in numbers
    #we will just call it by the by numbers for now
    #later, user will enter this in client side, and will simply make the list here
        target = dataFile[target_column]
        target = target._get_numeric_data()
        target = target.as_matrix()
        
        #set target_names equal to the values in the 3rd column
        target_names = list(dataFile.target_names)
        #remove all empty strings from the list
        target_names = filter(bool, target_names) 
        #target_names = array(target_names)

    #get rid of the Strings and turn into Numpy
	dataFile = dataFile._get_numeric_data()
        numpy_array = dataFile.as_matrix()
        #remove the index and target columns from the data
        numpy_array = numpy.delete(numpy_array, 1, axis=1)
        numpy_array = numpy.delete(numpy_array, 0, axis=1)

        #Generate the Deisicion Tree
        clf = DecisionTreeClassifier(max_depth=5)
        #run fit, which creates the decision tree based the data
        clf.fit(numpy_array, target)
        #run rules function to get JSON version of Decision tree from sklearn 
        JSONString = rules(clf, feature_names, target_names)
        #write JSON File with appropriate name to disk
        newName = csvFilePath.replace(' ', '')[:-4]
        fileName = newName + '_tree.json'
        json.dump(JSONString, open(fileName, 'wb'))
        return fileName


#create array of JSON objects to be accessed by front-end 
#returns fileName (and Path, it's in working directory)
def createJSONofData(csvFilePath):

    csvfile = open(csvFilePath, 'r')
    justName = csvFilePath.replace(' ', '')[:-4]
    fileName = justName +'_data.json'
    jsonfile = open(fileName, 'w')
    dataFile = pandas.read_csv(csvFilePath, header = 0, index_col=False, na_filter=False)

    #get feature names
    fieldnames = list(dataFile.columns.values)
    fieldnames = fieldnames[:-1]

    reader = csv.DictReader(csvfile, fieldnames)
    #skip the first row which are headers
    reader.next()

    #write out the file as an array of JS objects
    
    jsonfile.write('[')
    firstLine = reader.next()
    json.dump(firstLine, jsonfile)

    for row in reader:
        jsonfile.write(',')
        json.dump(row, jsonfile)
        jsonfile.write('\n')

    jsonfile.write(']')
    return fileName

#https://stackoverflow.com/questions/17126037/how-to-delete-only-the-content-of-file-in-python
def deleteContent(fName):
    with open(fName, "w"):
        pass

#MAIN
CSVFile = sys.argv[1]

#create JSON of tree from CSV
treeJSONName = createJSONTreefromData(CSVFile)
#create JSON of the data, so easily access in javascript (dataMatrix)
dataJSONName = createJSONofData(CSVFile)
#clear data in old index.html file
deleteContent('index.html')
#copy contents of template html to the new file
copyfile('generic_tree.html', 'index.html')
#edit the index.html to have the fileNames of both 
with open("index.html") as fp:
    soup = BeautifulSoup(fp, 'html.parser')


JSVars = '\nvar decisionTreeJSONFile' + ' = "' + treeJSONName + '";\n' + 'var dataJSON' + ' = "' + dataJSONName + '";\n' + 'var dataCSVFile' + ' = "' + CSVFile + '";'
tag = soup.new_tag('script')
tag.string = JSVars
soup.body.insert_before(tag)
html = soup.prettify(soup.original_encoding)
with open("index.html", "wb") as file:
    file.write(html)





    


