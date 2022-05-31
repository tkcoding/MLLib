import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
x_train = np.random.random((6400, 55, 2))
y_train = np.random.randint(2, size=(6400,1))

xx_train = np.array([x_train, x_train])
yy_train = np.array([y_train, y_train])

print(x_train.shape)
xx_train = np.vstack([x_train, x_train])
yy_train = np.vstack([y_train, y_train])

print(xx_train.shape)

Inputs = Input(shape=(2,, ))
hidden1 = Dense(units=100, activation="sigmoid")(Inputs)
hidden2 =  Dense(units=100, activation='relu')(hidden1)
predictions =  Dense(units=1, activation='sigmoid')(hidden2)

model = Model([Inputs], outputs=predictions)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(xx_train, yy_train, batch_size=10, epochs=5)



# def autoencoder(dims, act='relu'):
#     """
#     Fully connected auto-encoder model, symmetric.
#     Arguments:
#         dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
#             The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
#         act: activation, not applied to Input, Hidden and Output layers
#     return:
#         Model of autoencoder
#     """
#     n_stacks = len(dims) - 1
#     x = Input(shape=(dims[0],), name='input')
#     h = x

#     # internal layers in encoder
#     for i in range(n_stacks-1):
#         h = Dense(dims[i + 1], activation=act, name='encoder_%d' % i)(h)

#     # hidden layer
#     h = Dense(dims[-1], name='encoder_%d' % (n_stacks - 1))(h)  # hidden layer, features are extracted from here

#     # internal layers in decoder
#     for i in range(n_stacks-1, 0, -1):
#         h = Dense(dims[i], activation=act, name='decoder_%d' % i)(h)

#     # output
#     h = Dense(dims[0], name='decoder_0')(h)

#     return Model(inputs=x, outputs=h)


# autoencoder = autoencoder(xx_train.shape[-1])
# n_stacks = len(dims) - 1
# hidden = autoencoder.get_layer(name='encoder_%d' % (n_stacks - 1)).output
# self.encoder = Model(inputs=autoencoder.input, outputs=hidden)


# def pretrain(x, batch_size=256, epochs=200, optimizer='adam'):
#     print('...Pretraining...')
#     self.autoencoder.compile(loss='mse', optimizer=optimizer)  # SGD(lr=0.01, momentum=0.9),
#     self.autoencoder.fit(x, x, batch_size=batch_size, epochs=epochs)

# self.pretrain(x, batch_size)
# # self.autoencoder.save_weights('ae_weights.h5')
# # print('Pretrained weights are saved to ./ae_weights.h5')