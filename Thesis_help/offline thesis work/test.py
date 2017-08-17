#importing practice datasets from Sklearn! 
#values are in csv
#(values, feature Names, TargetEnumerator, TargetEnumeratorNames)
from sklearn.datasets import load_iris
from sklearn.datasets import load_linnerud

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
    else:
        feature = features[clf.tree_.feature[node_index]]
        threshold = clf.tree_.threshold[node_index]
        node['name'] = '{} > {}'.format(feature, threshold)
        left_index = clf.tree_.children_left[node_index]
        right_index = clf.tree_.children_right[node_index]
        node['children'] = [rules(clf, features, labels, right_index),
                            rules(clf, features, labels, left_index)]
    return node


def generateJSON(data):
	#create decisionTree object
    clf = DecisionTreeClassifier(max_depth=3)
    #run fit, which creates the decision based the data, and target attributes
    clf.fit(data.data, data.target)
    #run rules function to get JSON version of Decision tree from sklearn 
    JSONString = rules(clf, data.feature_names, data.target_names)
    #write JSON File with appropriate name to disk
    json.dump(JSONString, open(data.name +'.json', 'wb'))
    #Debugging Purposes
    print("generated " + data.name)
    return


#load the data sets
Iris_data = load_iris()
linnerud_data = load_linnerud()

#add name key to dictionarys to describe dataset for readability in the JSON when we try to output in d3
Iris_data['name'] = 'Iris'
linnerud_data['name'] = 'linnnerud'

#Decision Trees
generateJSON(Iris_data) #Decision Tree is fine (Decision Tree since descrete Set of Values as target)
generateJSON(linnerud_data)



    


