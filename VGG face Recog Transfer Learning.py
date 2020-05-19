#!/usr/bin/env python
# coding: utf-8

from keras.applications import VGG16

#downloading VGG16 weight
vggmodel = VGG16(weights = 'imagenet', include_top = False, input_shape = (224, 244, 3) )

#freezing all layer
for layer in vggmodel.layers:
    layer.trainable = False


# # creating Fully connected layers
def FClayer(model):
    newmodel = model.output
    newmodel = Flatten()(newmodel)
    newmodel = Dense(units = 256, activation = 'relu')(newmodel)
    newmodel = Dense(units = 128, activation = 'relu')(newmodel)
    newmodel = Dense(units = 32, activation = 'relu')(newmodel)
    newmodel = Dense(units = 2, activation = 'softmax')(newmodel)
    return newmodel


# # Adding FC onto VGG
from keras.layers import Dense, Flatten
from keras.models import Model

head = FClayer(vggmodel)

vggmodel = Model(inputs = vggmodel.input, outputs = head)

vggmodel.summary()


# ## Compiling model

# In[7]:


from keras.optimizers import RMSprop
vggmodel.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])


# ## Augmenting Images and traing the model
from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        './aftab/training/',
        target_size=(224, 244),
        class_mode='categorical')
test_set = test_datagen.flow_from_directory(
        './aftab/test/',
        target_size=(224, 244),
        class_mode='categorical')
vggmodel.fit(
        training_set,
        epochs=5,
        validation_data=test_set,
        )

vggmodel.save('vgg_face_recog.h5')
