import numpy as np
import io
from sklearn import preprocessing
from sklearn import svm
from sklearn.externals import joblib

filename = 'LinearSVCWithFeaturesAndShown.pkl'
train_slice = 0.7
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
    print arr
    return


def getInputData():
  f = open('train.csv')
  data = map(lambda line: line.split(','), f.readlines())
  purchased_data = filter(lambda line: len(line) == 25 and line[2] == '1', data)
  shown_data = filter(lambda line: len(line) == 25 and line[2] == '0', data)
  filtered_data = purchased_data * 10 + shown_data
  print len(filtered_data)
  tranformed_data = filter(lambda x: x, map(transform, filtered_data))
  print len(tranformed_data)
  return tranformed_data

def getPredictionInputData():
  f = open('test_v2.csv')
  data = map(lambda line: line.split(','), f.readlines())
  data = filter(lambda line: len(line) == 25, data)
  customer_features = {}
  for line in data:
    customer_features[line[0]] = line
  print len(customer_features)
  transformed_customer_features = {}
  for customer, features in customer_features.items():
    ans = transform(features)
    if not ans:
      print 'Transform failed on: ', features
      continue
    transformed_customer_features[customer] = ans
  print len(transformed_customer_features)
  return transformed_customer_features

def getTrainData(features):
  idx = int(train_slice * len(features))
  return features[:idx]

def getTestData(features):
  idx = int(train_slice * len(features))
  return features[idx:]

def removeElement(x, i):
  if (i == 0):
    return x[1:]
  if (i == 1):
    return [x[0]] + x[2:]
  return np.hstack((x[:i-1], x[i+1:]))

def trainModel(X, origX, i):
  print 'Building model for ', i
  newY = map(lambda x: x[i], origX)
  newX = map(lambda x: removeElement(x, i), X)
  clf = svm.LinearSVC()
  clf.fit(newX, newY)
  print 'Built model for ', i
  return clf

def testModel(X, origX, model, i):
  print 'Testing model for ', i
  Y = map(lambda x: x[i], origX)
  newX = map(lambda x: removeElement(x, i), X)
  print i, model.score(newX, Y)


features = getInputData()
customer_features = getPredictionInputData()
#features = getTrainData(features)
test_features = getTestData(features)
print len(features), len(test_features)

scaler = preprocessing.StandardScaler().fit(features)
X = scaler.transform(features)

for customer in customer_features:
  customer_features[customer] = scaler.transform(customer_features[customer])

models = []
models.append(trainModel(X, features, 0))
models.append(trainModel(X, features, 1))
models.append(trainModel(X, features, 2))
models.append(trainModel(X, features, 3))
models.append(trainModel(X, features, 4))
models.append(trainModel(X, features, 5))
models.append(trainModel(X, features, 6))
joblib.dump(models, filename) 
#models = joblib.load(filename)
# fo = open('result6.csv', 'w')
# fo.write('customer_ID,plan\n')
# for customer in customer_features:
#   line = customer + ','
#   for i in range(0, 7):
#     model = models[i]
#     input = customer_features[customer]
#     line += str(model.predict(removeElement(input, i))[0])
#   line += '\n'
#   fo.write(line)

testX = scaler.transform(test_features)
print testX
testModel(testX, test_features, models[0], 0)
testModel(testX, test_features, models[1], 1)
testModel(testX, test_features, models[2], 2)
testModel(testX, test_features, models[3], 3)
testModel(testX, test_features, models[4], 4)
testModel(testX, test_features, models[5], 5)
testModel(testX, test_features, models[6], 6)


