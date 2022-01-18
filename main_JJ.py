import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from datetime import datetime



def make_model(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)


    val_ds = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=0.2,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size)

    class_names = train_ds.class_names

    #Configure the dataset for performance
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    #Standardize the data
    normalization_layer = layers.Rescaling(1./255)

    normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    image_batch, labels_batch = next(iter(normalized_ds))
    first_image = image_batch[0]
    # Notice the pixel values are now in `[0,1]`.
    print(np.min(first_image), np.max(first_image))

    num_classes = len(class_names)

    data_augmentation = keras.Sequential(
        [
            layers.RandomFlip("horizontal",
                              input_shape=(img_height,
                                           img_width,
                                           3)),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
        ]
    )

    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
    ])


    model.load_weights('./saved_models/model_epk_1_date_18_01_2022__20_52_31.h5')
    print('Model Loaded!')

    model.compile(optimizer='adam',
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])



    return model, train_ds, val_ds, class_names

def training_results(model, epochs, train_ds, val_ds, checkpoint_path):


    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True
    )

    #Visualize training results

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[checkpoint_callback],
        epochs=epochs
    )

    model.save_weights('./saved_models/model_epk_'+str(epochs)+"_date_"+str(datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))+".h5")


    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

def main():
    #Creating a dataset
    batch_size = 32
    img_height = 150
    img_width = 150

    data_dir = 'Mushrooms'
    checkpoint_path = "data/model_checkpoint"

    epochs = 1

    #Creating new model
    model, train_ds, val_ds, class_names = make_model(data_dir, img_height, img_width, batch_size)

    #Loading model
    #model = tf.keras.models.load_model('./saved_model/my_model')


    model.summary()


    training_results(model, epochs, train_ds, val_ds,checkpoint_path)


    sunflower_url = "https://upload.wikimedia.org/wikipedia/commons/f/f0/Amanita_Muscaria_in_Eastern_Europe%2C_Lithuania.jpg"
    sunflower_path = tf.keras.utils.get_file('Pieczarka', origin=sunflower_url)

    img = tf.keras.utils.load_img(
        sunflower_path, target_size=(img_height, img_width)
    )
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(score)], 100 * np.max(score))
    )




if __name__ == "__main__":
    main()
