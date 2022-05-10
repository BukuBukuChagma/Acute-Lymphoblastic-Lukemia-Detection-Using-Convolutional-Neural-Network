from tensorflow.keras.models import  Sequential, save_model, load_model
from keras.layers import Flatten, Dense, MaxPooling2D, Conv2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
from keras import regularizers
from keras import callbacks
import tensorflow as tf

def intialize_model():
    model = tf.keras.models.Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(130, 130, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(64, activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    opt = SGD(learning_rate=0.001, momentum=0.9)
    model.compile(loss='binary_crossentropy',optimizer= opt,metrics =['accuracy'])
    return model


#get training and validation data
#train = image.ImageDataGenerator(rescale=1/255)
validation = image.ImageDataGenerator(rescale = 1/255)
#train_dataset = train.flow_from_directory('Dataset/processedTrain/', batch_size=20, target_size=(130, 130),class_mode='binary',)
validation_dataset = validation.flow_from_directory('Dataset/processedValidation/', batch_size=20,target_size=(130,130), class_mode='binary')
modelpath = "Model/"

##train the model
# model = intialize_model()
# early_stopping_monitor = callbacks.EarlyStopping(monitor='val_accuracy',patience=7,mode='auto',restore_best_weights=True)
# model_fit = model.fit(train_dataset, steps_per_epoch=len(train_dataset), epochs = 40, callbacks=[early_stopping_monitor], validation_data= validation_dataset, validation_steps=len(validation_dataset))
# save_model(model, modelpath)

#load the model to test accuracy on validation data
model = load_model(modelpath, compile = True)
_, acc = model.evaluate(validation_dataset, steps=len(validation_dataset))
print(acc * 100.0, '%')
