import numpy as np
import io
import math
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.externals.six import StringIO  
import pydot 

filename = 'DecisionTree.pkl'
train_slice = 0.7
output_file = 'result17.csv'
states = {
  'AK': 1,
  'AZ': 2,
  'AR': 3,
  'CA': 4,
  'CO': 5,
  'CT': 6,
  'DE': 7,
  'FL': 8,
  'GA': 9,
  'HI': 10,
  'ID': 11, 
  'IL': 12, 
  'IN': 13, 
  'IA': 14, 
  'KS': 15, 
  'KY': 16, 
  'LA': 17,
  'ME': 18, 
  'MD': 19, 
  'MA': 20, 
  'MI': 21, 
  'MN': 22, 
  'MS': 23, 
  'MO': 24, 
  'MT': 25, 
  'NE': 26,
  'NV': 27, 
  'NH': 28, 
  'NJ': 29, 
  'NM': 30, 
  'NY': 31, 
  'NC': 32, 
  'ND': 33, 
  'OH': 34, 
  'OK': 35, 
  'OR': 36, 
  'PA': 37, 
  'RI': 38, 
  'SC': 39, 
  'SD': 40, 
  'TN': 41, 
  'TX': 42, 
  'UT': 43, 
  'VT': 44,
  'VA': 45, 
  'WA': 46, 
  'WV': 47, 
  'WI': 48, 
  'WY': 49,
  'AL': 50,
  'DC': 51
};

car_value_dict = {
  'a': 1,
  'b': 2,
  'c': 3,
  'd': 4,
  'e': 5,
  'f': 6,
  'g': 7,
  'h': 8,
  'i': 9
}

def transform(arr):
  for i in range(0, 25):
    arr[i] = arr[i].strip()
  input = []
  if arr[11] == 'NA':
    arr[11] = '4'
  if arr[16] == 'NA':
    arr[16] = '0'
  if arr[15] == 'NA':
    arr[15] = '-1'
  if arr[10] == '':
    arr[10] = 'e'
  if arr[6] == 'NA':
    arr[6] = '10000'
  try:
    input.extend(map(lambda x: int(arr[x]), [17, 18, 19, 20, 21, 22, 23, 3, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 24]))
    parts = arr[4].split(':')
    if len(parts) != 2:
      print 'Parts is not 2', parts
      return
    input.append(int(parts[0]) * 60 + int(parts[1]))
    if (arr[5] not in states):
      print 'State not found', arr[5]
      return
    input.append(states[arr[5]])
    if (arr[10] not in car_value_dict):
      print 'car value not found', arr
      return
    input.append(car_value_dict[arr[10]])
    return input
  except ValueError:
    print 'Transform failed on: ', arr
    return

def getShoppingPoint(line):
  return int(line[1])

def getCost(line):
  return int(line[-1])

def getPurchase(line):
  return ''.join(line[-8:-1])

def getInputData():
  f = open('train.csv')
  data = map(lambda line: line.split(','), f.readlines())
  customer_lines = {}
  customer_results = {}
  customer_features = {}
  features = []
  results = []
  for line in data:
    if line[0] == 'customer_ID':
      continue
    if line[2] == '1':
      customer_results[line[0]] = getPurchase(line)
      continue
    if line[0] not in customer_lines:
      customer_lines[line[0]] = [line]
    else:
      customer_lines[line[0]].append(line)
    
  for customer, lines in customer_lines.items():
    customer_features[customer] = getFeaturesFromCustomer(lines)

  for customer in customer_lines:
    features.append(customer_features[customer])
    results.append(customer_results[customer])

  print 'Number of feature rows: ', len(features)
  print 'Number of result rows: ', len(results)
  print 'Sample Feature: ', features[0]
  print 'Sample result: ', results[0]
  return (features, results)

def getFeaturesFromCustomer(lines):
  feature = []
  for line in lines:
    transformed_line = transform(line)
    if not transformed_line:
      continue
    
    elif not feature:
      feature = transformed_line[7:]
    else:
      feature.append(int(getPurchase(line)))
    if not feature:
      continue
  while len(feature) < 50:
    feature.extend([-1, -1, -1, -1, -1, -1, -1])
  return feature

def getPredictions(model, scaler):
  f = open('test_v2.csv')
  data = map(lambda line: line.split(','), f.readlines())
  data = filter(lambda line: len(line) == 25, data)
  input = {}
  customer_features = {}
  result = {}
  for line in data:
    if line[0] == 'customer':
      continue
    customer = line[0]
    transformed_line = transform(line)
    if not transformed_line or len(transformed_line) < 7:
      print 'Bad transformation: ', transformed_line
      continue
    if customer not in customer_features:
      input[customer] = [line]
    else:
      input[customer].append(line)

  for customer, lines in input.items():
    customer_features[customer] = getFeaturesFromCustomer(lines)
  
  THRESHOLD = 5
  for customer in customer_features:
    scaled_features = scaler.transform(customer_features[customer])
    prediction = model.predict(scaled_features)
    result[customer] = prediction

  print 'Number of result rows: ', len(result)
  print 'Sample Result: ', result.values()[0]
  return result

def getTrainData(features, results):
  idx = int(train_slice * len(features))
  return (features[:idx], results[:idx])

def getTestData(features, results):
  idx = int(train_slice * len(features))
  return (features[idx:], results[idx:])

def removeElement(x, i):
  if (i == 0):
    return x[1:]
  if (i == 1):
    return [x[0]] + x[2:]
  return np.hstack((x[:i-1], x[i+1:]))

def trainModel(X, results):
  print 'Building model...'
  clf = tree.DecisionTreeClassifier()
  clf.fit(X, results)
  return clf

def testModel(X, results, model):
  print 'Testing model'
  print model.score(X, results)

def buildVisualization(model):
  dot_data = StringIO() 
  tree.export_graphviz(model, out_file=dot_data) 
  graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
  graph.write_pdf("iris.pdf") 

(features, results) = getInputData()
scaler = preprocessing.StandardScaler().fit(features)
# X = scaler.transform(features)

(train_features, train_results) = getTrainData(features, results)
(test_features, test_results) = getTestData(features, results)
print len(train_features), len(test_features)
X = scaler.transform(train_features)
model = trainModel(X, train_results)
model = trainModel(X, results)
# buildVisualization(model)

joblib.dump(model, filename) 
# model = joblib.load(filename)
# fo = open(output_file, 'w')
# fo.write('customer_ID,plan\n')
# results = getPredictions(model, scaler)
# for customer in results:
#   line = customer + ',' + results[customer] + '\n'
#   fo.write(line)

testX = scaler.transform(test_features)
testModel(testX, test_results, model)