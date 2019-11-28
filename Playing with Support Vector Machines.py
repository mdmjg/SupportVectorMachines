#!/usr/bin/env python
# coding: utf-8

# In[39]:


one = [(2, 2), (4, 4), (4, 0)] 
two =  [(0, 0), (2, 0), (0, 2)] 
plt.figure(figsize = (10, 10)) 
for x, y in one: 
    plt.scatter(x, y, c = 'g')    
for x, y in two: 
    plt.scatter(x, y, c = 'b')  
plt.scatter(one[0], one[1], c = 'g', label = '+1')
plt.scatter(two[0], two[1], c = 'b', label = '-1')
# the line -x + 3 separates the datapoints
plt.plot(np.linspace(0, 4, 100), [-i +3 for i in np.linspace(0, 4, 100)], '--', linewidth=2,)
plt.xlabel('$x_1$') 
plt.ylabel('$x_2$') 
plt.legend()


# By inspection, we can see our hyperplane is x1 + w2 - 3 = 0
# 

# In[40]:


plt.figure(figsize = (10, 10)) 
for x, y in one: 
    plt.scatter(x, y, c = 'g')    
for x, y in two: 
    plt.scatter(x, y, c = 'b')  
plt.scatter(one[0], one[1], c = 'g', label = '+1')
plt.scatter(two[0], two[1], c = 'b', label = '-1')
# the line -x + 3 separates the datapoints
plt.plot(np.linspace(0, 4, 100), [-i +3 for i in np.linspace(0, 4, 100)], '--', linewidth=2,)
plt.plot(np.linspace(0, 4, 100), [-i +2 for i in np.linspace(0, 4, 100)], '--', linewidth=2,)
plt.plot(np.linspace(0, 4, 100), [-i +4 for i in np.linspace(0, 4, 100)], '--', linewidth=2,)
plt.xlabel('$x_1$') 
plt.ylabel('$x_2$') 
plt.legend()


# In[54]:


import collections
import numpy.matlib 
import numpy as np 
f1 = open("spam_train.txt", "r")
contents =f1.readlines()
training = []
validating = []
isSpam = []


for i in range(len(contents)):
    if i < 4000:
        training.append(contents[i])
    else:
        validating.append(contents[i])
        
        
def vectorize_email(notIgnoredVocab, data):
    y = []
    vectorized = []
    x = []
    for email in data:
        emailDict = dict()
        words = email.split()
        for i in range(len(words)): # iterate over each word
            if i == 0: 
                if words[i] == '0':
                    y.append(-1)
                else:
                    y.append(1)
            else:
                emailDict[words[i]] = 1
        vectorized.append(emailDict)
    
    
    for email in vectorized:
        ind_x = []
        for i in range(len(notIgnoredVocab)):
            if notIgnoredVocab[i] in email:
                ind_x.append(1)
            else:
                ind_x.append(0)
        x.append(np.array(ind_x))

    return np.array(x), y


def getVocab():
    vocab = dict()
    for email in training:
        emailDict = dict()
        words = email.split()
        for i in range(1, len(words)):
            # is this the first time we see the word in this email?
            if words[i] not in emailDict:
                emailDict[words[i]] = 1
                if words[i] not in vocab:
                    vocab[words[i]] = 1
                else:
                    vocab[words[i]] += 1
            
            else:
                
                emailDict[words[i]] = 1
    return vocab
    


def getNotIgnoredVocab():
    notIgnoredVocab = list()
    vocab = getVocab()
    for key in vocab:
        if vocab[key] >= 30:
            notIgnoredVocab.append(key)
    
    return notIgnoredVocab



notIgnoredVocab = getNotIgnoredVocab()
print(len(notIgnoredVocab))
x_train, y_train = vectorize_email(notIgnoredVocab, training)
print(len(y_train))
print(len(x_train[0]))
print(y_train[0])
x_test, y_test = vectorize_email(notIgnoredVocab, validating)
print(len(x_test))
    


# In[82]:


def pegasos_svm_train(x,y,lam,w=None):
    if w == None:
        w = np.zeros(len(x[0]))
    num_instances = len(x)
    objectives = list()
    t = 0
    for epoch in range(20):
        for i in range(num_instances):
            t += 1
            step = 1/(t*lam)
            dot = np.dot(w, x[i])
            if y[i]*dot < 1: 
                w = (1 - step*lam)*w + step*y[i]*x[i]

            else: 
                w = (1 - step*lam)*w 

        objectives.append(lam/2*np.linalg.norm(w)**2 + 1/num_instances*sum([max(0, 1 - y[j]*np.dot(w, x[j])) for j in range(num_instances)]))
                
    return w, objectives


# In[83]:


w, objectives = pegasos_svm_train(x_train, y_train, 2**-5)
plt.figure(figsize = (8, 5))
plt.plot(range(20), objectives) 
plt.xlabel('Iteration') 
plt.ylabel('Objective')


# #### 4.2.2

# In[84]:


def pegasos_svm_test(x, y, w):
    numMistakes = 0
    numIterations = 0
    for i in range(len(x)):
        dot = np.dot(w, x[i])
        dot = 1 if dot >= 0 else 0
        if y[i] != dot: 
            numMistakes += 1
        numIterations += 1
    return float(numMistakes)/numIterations


# In[85]:


w, objectives = pegasos_svm_train(x_train, y_train,2**-5)
print(pegasos_svm_test(x_train, y_train, w))
print(pegasos_svm_test(x_test, y_test, w))


# In[86]:


def hinge_loss(x, y, w):
    return 1 / len(x) * sum([max(0, 1 - y[i]*np.dot(w, x[i])) for i in range(len(y))])


# In[87]:


# run learning algo on different values of lambda
expo = np.arange(9, -2, -1)
lambs = []
 

lambs = [2**-9,2**-8, 2**-7, 2**-6, 2**-5, 2**-4, 2**-3, 2**-2, 2**-1, 1, 2]
weights = []
hinges = []
training_error = []
validation_error = []
for l in lambs:
    w, _ = pegasos_svm_train(x_train, y_train, l)
    weights.append(w)
    hinges.append(hinge_loss(x_train, y_train,w))
    training_error.append(pegasos_svm_test(x_train, y_train, w))
    validation_error.append(pegasos_svm_test(x_test, y_test, w))

plt.plot(np.log2(lambs), training_error, label = 'Training Error') 
plt.plot(np.log2(lambs), hinges, label = 'Training Hinge Loss') 
plt.plot(np.log2(lambs), validation_error, label = 'Validation Error') 
plt.legend()


print("For the classifier that has the smallest validate error, test error = ", )
best_w = weights[validation_error.index(min(validation_error))]

    


# #### 4.2.4i The test error for the hinge loss is 

# In[88]:


print("Minimum validation error ", min(validation_error))


# #### 4.2.4ii

# In[89]:


print("Classifier with smallest validation error", pegasos_svm_test(x_test, y_test, best_w))


# In[90]:


def support_vectors(x, w):
    distance = min(abs(np.dot(w, np.array(x).transpose())))
    num_vectors = len([1 for i in x if np.dot(w, i) == distance])
    print("Number of support vectors: ", num_vectors) 

