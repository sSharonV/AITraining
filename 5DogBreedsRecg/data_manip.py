import turicreate as tc
import os

#For string input it interpreters as a file
data = tc.SFrame("dog_classifier.sframe")

"""
Randomly split the rows of an SFrame into two SFrames
-  The first SFrame contains M rows, sampled uniformly (without replacement) from the original SFrame
"""
testing, training = data.random_split(0.8)

"""
image_classifier params: target(string\int) is the column which contains the target variable
                         model(specific strings) indicates the which ML algo would be used
resnet-50(one of the most accurate machine learning model architectures)
    -> https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33   
- This automatically training the model according to 'testing' dataset                       
"""

classifier = tc.image_classifier.create(testing, target="label", model="resnet-50")

#To evaulate 
testing = classifier.evaluate(training)

"""
Export trained model
'save(str)' used for later performing 'load_model()'
export_coreml is for high-class use of the model (apple uses CoreML in apps)
"""
classifier.save("dog_classifier.model")
classifier.export_coreml("dog_classifier.mlmodel")
print ("Done training and saved dog_classifier.model")
print testing["accuracy"]