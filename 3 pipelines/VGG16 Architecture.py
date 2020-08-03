import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.multiclass import unique_labels
import matplotlib.cm as cm
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from google.colab import drive
drive.mount('/content/drive/', force_remount=True)
X = pickle.load(open("/content/drive/My Drive/Big Data Final/X.pickle", "rb"))
y = pickle.load(open("/content/drive/My Drive/Big Data Final/y.pickle", "rb"))

X = X/255.0  # normalise

(trainX, testX, trainY, testY) = train_test_split(X, y, test_size=0.20, random_state=42)
model = tf.keras.Sequential()

#layer 1
model.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=X.shape[1:], padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization())

#Layer 2
model.add(tf.keras.layers.Conv2D(64, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())

#Layer 3
model.add(tf.keras.layers.Conv2D(128, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization())

#layer 4
model.add(tf.keras.layers.Conv2D(128, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())

#Layer 5
model.add(tf.keras.layers.Conv2D(256, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization())

#Layer 6
model.add(tf.keras.layers.Conv2D(256, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization())

#Layer 7
model.add(tf.keras.layers.Conv2D(256, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())

#Layer 8
model.add(tf.keras.layers.Conv2D(512, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization())

#Layer 9
model.add(tf.keras.layers.Conv2D(512, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization())

#Layer 10
model.add(tf.keras.layers.Conv2D(512, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())

#Layer 11
model.add(tf.keras.layers.Conv2D(512, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization())

#Layer 12
model.add(tf.keras.layers.Conv2D(512, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.BatchNormalization())

#Layer 13
model.add(tf.keras.layers.Conv2D(512, (3, 3),  padding="same"))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same'))
model.add(tf.keras.layers.BatchNormalization())

#Layer 14
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(4096))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

#Layer 15
model.add(tf.keras.layers.Dense(4096))
model.add(tf.keras.layers.Activation("relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.BatchNormalization())

#Layer 16
model.add(tf.keras.layers.Dense(1))
model.add(tf.keras.layers.Activation('sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
Modes = model.fit(trainX, trainY, batch_size=15, epochs=100, validation_data=(testX, testY))

#Saving the model
model.save('/content/drive/My Drive/Big Data Final/VGG16.model')
predictions = model.predict(testX)
sport = ["Successful", "Unsuccessful"]

#Results (Classification Report)
print(classification_report(testY, predictions, target_names=sport))

#Results (Validation, Accuracy) [Skelearn]
acc = Modes.Modes['acc']
val_acc = Modes.Modes['val_acc']
loss = Modes.Modes['loss']
val_loss = Modes.Modes['val_loss']
epochs = range(len(acc))
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()

#Print Roc Curves
predict_proba = model.predict_proba(testX)
auc = roc_auc_score(testY, predict_proba)
fpr, tpr, _ = roc_curve(testY, predict_proba)
plt.plot(fpr, tpr, label="CNN - VGG16, auc="+str(auc))
plt.plot([0, 1], [0, 1], 'r--')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#print Confusion Matrix
ls = list()
class_name = unique_labels(testY, predictions)
print(class_name)
for i in class_name:
  a = sport[int(class_name[0])]
  b = sport[int(class_name[1])]
ls.append(a)
ls.append(b)
class_names = ls


def plot_confusion_matrixx(y_true, y_pred, classes, normalize=False, title=None, cmap=cm.Blues):

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ax.plot(range(1), range(1))

    ax.set_aspect('equal')
    return ax


plot_confusion_matrixx(testY, predictions, classes=class_names,
                       title='(VGG16) Confusion matrix, without normalization')
plt.show()
