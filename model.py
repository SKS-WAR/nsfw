from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255) # rescale the pixel values to between 0 and 1
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        'data/test/',
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

# step 2

from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(5, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# step 3

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.n // train_generator.batch_size,
        epochs=10,
        validation_data=test_generator,
        validation_steps=test_generator.n // test_generator.batch_size)

# step 4

score = model.evaluate_generator(test_generator, steps=test_generator.n // test_generator.batch_size)

print('Test loss:', score[0])
print('Test accuracy:', score[1])


# step 5

import numpy as np

test_generator.reset()
preds = model.predict_generator(test_generator, steps=test_generator.n // test_generator.batch_size + 1)
predicted_classes = np.argmax(preds, axis=1)

class_names = ['drawings', 'hentai', 'neutral', 'porn', 'sexy']

for i in range(len(predicted_classes)):
    print(f"Predicted class: {class_names[predicted_classes[i]]}")


# save model

model.save('image_classifier.h5')
