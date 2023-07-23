import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# Directory paths for the training set and the testing set

#6400 images in training sets
train_data_dir = '/Users/robertcalistri/Downloads/BinaryClassifierTestTrainData/train'
#1800 images in testing sets
test_data_dir = '/Users/robertcalistri/Downloads/BinaryClassifierTestTrainData/test'

# Declare image dimensions and the batch size to be processed
img_width, img_height = 299, 299
batch_size = 32

# Number of KFolds to be implemented
num_folds = 4

# Creates an image data generator for training data augmentation
# - rescale: Rescales each image by 1/255
# - Shear range: Angle at which images are sheared
# - Horizontally flips image: Beneficial for lung x-rays as lungs have a vertical
# plane of symmetry, but are typically asymmetric across the horizontal plane
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Creates image data generator for the test data, simply resizes images to
# match the image size used in the training data generator
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

# Feeds images from directory into generator
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Feeds images from directory into generator
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# Combine train and test data for k-fold cross-validation
combined_data = np.vstack((train_generator[0][0], test_generator[0][0]))
combined_labels = np.hstack((train_generator[0][1], test_generator[0][1]))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

# Lists to store accuracy and loss for each fold
fold_accuracies = []
fold_losses = []


# Define the sequential layers of the CNN:
# - 3 convoluted layers
# - 3 pooling layers
# - Flatten the data
# - 2 Dense layers
# - 1 Dropout layer to prevent overfitting
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model for use
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

for fold, (train_index, val_index) in enumerate(kf.split(combined_data)):
    print(f"Fold {fold+1}/{num_folds}")
    
    # Create data generators for the current fold
    train_data_fold, val_data_fold = combined_data[train_index], combined_data[val_index]
    train_labels_fold, val_labels_fold = combined_labels[train_index], combined_labels[val_index]

    train_generator_fold = train_datagen.flow(
        train_data_fold,
        train_labels_fold,
        batch_size=batch_size
    )

    val_generator_fold = train_datagen.flow(
        val_data_fold,
        val_labels_fold,
        batch_size=batch_size
    )

    # Train the model
    # This model uses 10 epochs, with each epoch having 195 steps
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=10,
        validation_data=test_generator,
        validation_steps=test_generator.samples // batch_size
    )
    
    # Evaluate the model on test data for this fold
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f"Test accuracy for fold {fold+1}: {test_accuracy}")

    fold_accuracies.append(test_accuracy)
    fold_losses.append(test_loss)

# Calculate and print the average accuracy and loss across all folds
avg_accuracy = np.mean(fold_accuracies)
avg_loss = np.mean(fold_losses)
print("Average accuracy across all folds:", avg_accuracy)
print("Average loss across all folds:", avg_loss)

# Saves model as a file
model.save('binaryClassifier.h5')

# Plot the accuracy and loss through each epoch
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test accuracy: {test_accuracy}")

# Make predictions on test data
predictions = model.predict(test_generator)
y_pred = np.round(predictions).flatten()
y_true = test_generator.classes

# Generate classification report and confusion matrix
print("Classification Report:")
print(classification_report(y_true, y_pred))

confusion_mat = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(confusion_mat)
