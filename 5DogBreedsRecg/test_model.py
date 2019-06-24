import turicreate as tc
import os

train_model = tc.load_model('dog_classifier.model')
image_test = tc.load_images('newdog.jpg')
image_test['predictions'] = train_model.predict(image_test)
print(image_test['predictions'])
print(image_test)