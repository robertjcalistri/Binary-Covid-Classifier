import tensorflow as tf
from PIL import Image
from keras.utils import img_to_array, load_img
import numpy as np

model = tf.keras.models.load_model('binaryClassifier.h5')

test_img_covid = load_img('/Users/robertcalistri/Downloads/BinaryClassifierTestTrainData/test/COVID/COVID-343.png')
test_img_covid = tf.image.resize(test_img_covid,(150,150), method='nearest')

test_img_normal = load_img('/Users/robertcalistri/Downloads/BinaryClassifierTestTrainData/test/normal/Normal-5990.png')
test_img_normal = tf.image.resize(test_img_normal,(150,150), method='nearest')

test_img_covid = img_to_array(test_img_covid)
test_img_normal = img_to_array(test_img_normal)

test_img_covid = np.expand_dims(test_img_covid, axis=0)
test_img_normal = np.expand_dims(test_img_normal, axis=0)

result_covid = model.predict_classes(test_img_covid)
result_normal = model.predict_classes(test_img_normal)

print(result_covid)
print(result_normal)