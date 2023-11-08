def CNN(X_train, y_train, X_test, y_test, epochs=10, output=10):
    import tensorflow as tf
    from tensorflow import keras
    from keras import Sequential
    from keras.layers import (
        Dense,
        Flatten,
        Dropout,
        Activation,
        Conv2D,
        MaxPooling2D,
        BatchNormalization,
    )
    from keras.optimizers import RMSprop
    from keras.callbacks import ReduceLROnPlateau, EarlyStopping
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    import numpy as np
    from keras.callbacks import ModelCheckpoint
    from keras.preprocessing.image import ImageDataGenerator
    from keras.utils import to_categorical
    from datetime import datetime

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=10,
        width_shift_range=0.25,
        height_shift_range=0.25,
        shear_range=0.1,
        zoom_range=0.25,
        horizontal_flip=False,
    )

    valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    # y_train = to_categorical(y_train, num_classes=output)
    # y_test = to_categorical(y_test, num_classes=output)

    model = Sequential()

    # # 1st convolutional Layer
    # model.add(Conv2D(64, (3, 3), input_shape=X_train.shape[1:]))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # # 2st convolutional Layer
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # # 3st convolutional Layer
    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation("relu"))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # num_conv_layers = 3  # Number of convolutional layers
    # num_filters = [32, 64, 128, 64, 32]  # Number of filters for each layer
    # kernel_size = (5, 5)  # Kernel size for each layer
    # pool_size = (2, 2)  # Pooling size for each layer

    # # Loop to add convolutional layers
    # for i in range(num_conv_layers):
    #     if i == 0:
    #         # For the first layer, specify the input shape
    #         model.add(
    #             Conv2D(
    #                 num_filters[i],
    #                 kernel_size,
    #                 activation="relu",
    #                 input_shape=X_train.shape[1:],
    #             )
    #         )
    #     else:
    #         model.add(Conv2D(num_filters[i], kernel_size, activation="relu"))

    #     model.add(MaxPooling2D(5))  # pool_size=pool_size

    # # Fully Connected Layer
    # model.add(Flatten())
    # model.add(Dense(1000, activation="relu"))
    # model.add(Dense(512, activation="relu", use_bias=True))
    # model.add(Dense(256, activation="relu", use_bias=True))
    # model.add(Dropout(0.25))
    # model.add(Dense(128, activation="relu", use_bias=True))
    # model.add(Dense(64, activation="relu", use_bias=True))
    # model.add(Dense(32, activation="relu", use_bias=True))
    # model.add(Dense(output, activation="softmax"))

    model.add(Conv2D(64, kernel_size=3, activation="relu", input_shape=(100, 100, 1)))
    model.add(BatchNormalization())  # ----------------
    model.add(Conv2D(64, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())  # ----------------
    model.add(Conv2D(64, kernel_size=5, padding="same", activation="relu"))
    model.add(BatchNormalization())  # ----------------
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))  # ----------------

    model.add(Conv2D(128, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, kernel_size=5, padding="same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(256, kernel_size=3, activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1000, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dense(256, activation="relu", use_bias=True))
    model.add(BatchNormalization())
    model.add(Dense(128, activation="relu", use_bias=True))
    model.add(BatchNormalization())
    model.add(Dense(64, activation="relu", use_bias=True))
    model.add(BatchNormalization())
    model.add(Dense(output, activation="softmax"))

    print(model.summary())
    # print(len(X_train))

    learning_rate = 0.001
    optimizer = RMSprop(learning_rate=learning_rate)
    learning_rate_reduction = ReduceLROnPlateau(
        monitor="loss", patience=200, verbose=1, factor=0.2
    )

    ch = ModelCheckpoint(
        "models/facial_model.h5",
        monitor="accuracy",
        verbose=0,
        save_best_only=True,
        mode="max",
    )

    # es = EarlyStopping(monitor="loss", mode="min", verbose=0, patience=200)
    es = EarlyStopping(
        monitor="val_acc", mode="auto", verbose=1, baseline=0.90, patience=0
    )

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir="logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # model.compile(
    #     loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    # )

    # epochs = 50
    # batch_size = 20
    # print(X_train.shape)
    # history = model.fit(
    #     train_datagen.flow(X_train, y_train, batch_size=batch_size),
    #     steps_per_epoch=X_train.shape[0] // batch_size,
    #     epochs=epochs,
    #     validation_data=valid_datagen.flow(X_test, y_test),
    #     shuffle="True",
    #     validation_split=0.3,
    #     validation_steps=50,
    #     verbose=1,
    #     callbacks=[learning_rate_reduction, es, ch, tensorboard_callback],
    # )

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        shuffle="True",
        validation_data=(X_test, y_test),
        validation_split=0.3,
        callbacks=[learning_rate_reduction, es, ch, tensorboard_callback],
    )

    # loss, acc = model.evaluate(valid_datagen.flow(X_test, y_test))

    # print(f"Loss: {loss}\nAccuracy: {acc*100}")

    result = model.evaluate(X_test, y_test, batch_size=10)

    print("test Loss : ", result[0], " , Test Accuracy : ", result[1])

    return model


# (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# # Normalize because rgb values are ranging from 0 to 255

# X_train = tf.keras.utils.normalize(X_train, axis=1)
# X_test = tf.keras.utils.normalize(X_test, axis=1)
# plt.imshow(X_train[0], cmap=plt.cm.binay)

# print("training set : ", X_train.shape, "  Dimention od each image:", X_train[0].shape)

# IMG_SIZE = 28

# X_trainr = np.array(X_train).reshape(-1, IMG_SIZE, 1)
# X_trainr = np.array(X_test).reshape(-1, IMG_SIZE, 1)


# if len(x_train[0].shape) > 1:
#     print("Flattning....")
#     model.add(Flatten(input_shape=x_train[0].shape))

# output_range = int(y_train[y_train.argmax(axis=0)]) + 1
# print("range: ", output_range, " y : ", y_train)
