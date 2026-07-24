import os

from keras.src.layers import BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
# import tensorflow as tf

DATASET_PATH='dataset/'
print(os.listdir(DATASET_PATH))
# num_classes=len(os.listdir(DATASET_PATH))
num_classes=len([i for i in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, i))])
class_mode='binary' if num_classes ==2  else "categorical"

train_datagen=ImageDataGenerator(rescale=1./255,
                                 validation_split=0.3,
                                 rotation_range=30,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 zoom_range=0.25,
                                 horizontal_flip=True,
                                 shear_range=0.2,
                                 brightness_range=[0.8,1.2],
                                 fill_mode='nearest'
)

val_datagen=ImageDataGenerator(
    rescale=1./255,
    validation_split=0.3,
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

train_data=train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(160, 160),
    batch_size=32,
    class_mode=class_mode,
    subset="training",
    shuffle=True
)

val_data=val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(160, 160),
    batch_size=32,
    class_mode=class_mode,
    subset="validation",
    shuffle=False
)

reduce_lr=ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-4
)

model=Sequential([
    Input(shape=(160, 160, 3)),
    Conv2D(16, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu', padding="same"),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    GlobalAveragePooling2D(),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid') if class_mode == 'binary'
        else Dense(num_classes, activation='softmax')
])

loss_function='binary_crossentropy' if class_mode=='binary' else 'categorical_crossentropy'

model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
model.summary()
print(train_data.class_indices)
class_names=list(train_data.class_indices.keys())

model.fit(train_data, validation_data=val_data, epochs=10, callbacks=[early_stop, reduce_lr])

test_data, test_accuracy=model.evaluate(val_data)
print(test_data)
print(f'Точность модели на валидационных данных: {test_accuracy:.2f}')
model.save('image_classifier.keras')

