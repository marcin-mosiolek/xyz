import keras

from dataset import DataLoader
import model

data = DataLoader("../../autencoder/small.npy")

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
callbacks_list = [earlystop]

model.autoencoder.fit(
    data.train_x, data.train_y, nb_epoch=500, batch_size=64,
    shuffle=True, validation_data=(data.valid_x, data.valid_y), verbose=1, callbacks=callbacks_list)
