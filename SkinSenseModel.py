# -*- coding: utf-8 -*-

import os


train_normal_dir = os.path.join('./Training/Normal')


train_acne_dir = os.path.join('./Training/Acne')


validation_normal_dir = os.path.join('./Validation/Normal')


validation_acne_dir = os.path.join('./Validation/Acne')


import tensorflow as tf

model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
   
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Flatten(),
  
    tf.keras.layers.Dense(512, activation='relu'),

    tf.keras.layers.Dense(1, activation='sigmoid')
])



model.summary()



from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])



from tensorflow.keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(rescale=1/255,)
validation_datagen = ImageDataGenerator(rescale=1/255)


train_generator = train_datagen.flow_from_directory(
        './Training', 
        target_size=(300, 300),
        batch_size=60,
        
        class_mode='binary')


validation_generator = validation_datagen.flow_from_directory(
        './Validation',  
        target_size=(300, 300), 
        batch_size=20,
       
        class_mode='binary')



history = model.fit(
      train_generator,
      steps_per_epoch=20,  
      epochs=5,
      verbose=1,
      validation_data = validation_generator,
      validation_steps= 20)





