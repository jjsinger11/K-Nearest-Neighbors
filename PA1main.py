import numpy as np
from numpy import genfromtxt
from numpy import linalg

# Read Data Sets
train_data = genfromtxt('Data Files/pa1train.txt', delimiter=' ')
validate_data = genfromtxt('Data Files/pa1validate.txt', delimiter=' ')
test_data = genfromtxt('Data Files/pa1test.txt', delimiter=' ')
projection_data = genfromtxt('Data Files/projection.txt', delimiter=' ')


# Find the most common label with the shortest distance to the feature vector
def closestLabel(data, k, trainData):
    numRows = len(trainData)
    # Initialize k values in the array to 0
    smallestDist = np.zeros((k, 2))
    allDist = np.zeros(len(trainData))

    # Store all the distances of the feature vectors to the input label
    for i in range(numRows):
        a = np.array(data)
        a[len(a) - 1] = 0
        b = np.array(trainData[i])
        b[len(b) - 1] = 0
        allDist[i] = np.linalg.norm(a - b)

    # Temporarily fill k spots with the first distances
    for populate in range(k):
        smallestDist[populate][0] = allDist[populate]
        smallestDist[populate][1] = trainData[populate][len(trainData[0]) - 1]
    # Get the smallest elements of the distance array
    for i in range(k, len(allDist)):
        # Returns the maxValue of the columns
        maxValue = np.amax(smallestDist, axis=0)[0]
        # Find the feature vectors with the smallest distance to the input vector and store their
        # distances and labels
        if allDist[i] < maxValue:
            result = np.where(smallestDist == maxValue)
            smallestDist[result[0][0]][0] = allDist[i]
            smallestDist[result[0][0]][1] = trainData[i][len(trainData[0]) - 1]

    # Find the Most Frequent Label
    labelArr = np.zeros(len(smallestDist))
    # Store all the labels in an array
    for label in range(len(smallestDist)):
        labelArr[label] = smallestDist[label][1]
    labelArr = labelArr.astype(int)
    counts = np.bincount(labelArr)
    finalNum = np.argmax(counts)
    return finalNum


# Given the data sets and k, returns the percent error in the algorithm
def totalError(trainData, testData, k):
    numErrors = 0
    # Compare the vector against all other vectors and see if it can correctly identify it's label
    for vector in range(len(testData)):
        vectorLabel = testData[vector][len(testData[0]) - 1]
        label = closestLabel(testData[vector], k, trainData)
        if label != vectorLabel:
            numErrors += 1
    return numErrors / len(testData)


# Compresses the image files by using multiplying the image with the projection matrix
# Returns the error after running the k-nearest neighbors algorithm
def compressImage(trainData, testData, k):
    # Stores the labels of the data sets
    labelsTrain = trainData[:, -1]
    columnLabelsTrain = np.reshape(labelsTrain, (-1, 1))
    labelsTest = testData[:, -1]
    columnLabelsTest = np.reshape(labelsTest, (-1, 1))

    # Removes the last column from the data, leaving only the pixel values
    imageMatrixTrain = trainData[:, :-1]
    imageMatrixTest = testData[:, :-1]

    # Multiplies the data set with the projection and then adds the labels back
    compressTrain = np.matmul(imageMatrixTrain, projection_data)
    compressTrain = np.append(compressTrain, columnLabelsTrain, 1)
    compressTest = np.matmul(imageMatrixTest, projection_data)
    compressTest = np.append(compressTest, columnLabelsTest, 1)
    return totalError(compressTrain, compressTest, k)


k = 9
print("Algorithm for", k, "nearest neighbor(s)")
print("Percentage error:", totalError(train_data, train_data, k))
