import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Lambda, Embedding, Input, Concatenate, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

PATH = './ml-latest-small/'

ratings = pd.read_csv(PATH + 'ratings.csv')
print(ratings[:15])

movies = pd.read_csv(PATH + 'movies.csv')
print(movies.head())

g = ratings.groupby('userId')['rating'].count()
top_users = g.sort_values(ascending=False)[:15]

g = ratings.groupby('movieId')['rating'].count()
top_movies = g.sort_values(ascending=False)[:15]

top_r = ratings.join(top_users, rsuffix='_r', how='inner', on='userId')
top_r = top_r.join(top_movies, rsuffix='_r', how='inner', on='movieId')

print(pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum))

user_enc = LabelEncoder()
ratings['user'] = user_enc.fit_transform(ratings['userId'].values)
n_users = ratings['user'].nunique()

item_enc = LabelEncoder()
ratings['movie'] = item_enc.fit_transform(ratings['movieId'].values)
print(ratings['movie'])
n_movies = ratings['movie'].nunique()
min_rating = min(ratings['rating'])
max_rating = max(ratings['rating'])

print(n_users, n_movies, min_rating, max_rating)

X = ratings[['user', 'movie']].values
y = ratings['rating'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

n_factors = 50

X_train_array = [X_train[:, 0], X_train[:, 1]]
X_test_array = [X_test[:, 0], X_test[:, 1]]


class EmbeddingLayer:
    def __init__(self, n_items, n_factors):
        self.n_item = n_items
        self.n_factors = n_factors

    def __call__(self, x):
        x = Embedding(self.n_item, self.n_factors, embeddings_initializer='he_normal',
                      embeddings_regularizer=l2(1e-6))(x)
        x = Reshape((self.n_factors,))(x)
        return x


user = Input(shape=(1,))
u = EmbeddingLayer(n_users, n_factors)(user)

movie = Input(shape=(1,))
m = EmbeddingLayer(n_movies, n_factors)(movie)

x = Concatenate()([u, m])
x = Dropout(0.05)(x)

x = Dense(10, kernel_initializer='he_normal')(x)
x = Activation('relu')(x)
x = Dropout(0.5)(x)

x = Dense(1, kernel_initializer='he_normal')(x)
x = Activation('sigmoid')(x)
x = Lambda(lambda x: x * (max_rating - min_rating) + min_rating)(x)

model = Model(inputs=[user, movie], outputs=x)
opt = Adam(lr=0.001)
model.compile(loss='mean_squared_error', optimizer=opt)

print(model.summary())

history = model.fit(x=X_train_array, y=y_train, batch_size=64, epochs=5, verbose=1,
                    validation_data=(X_test_array, y_test))

plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnidute")
plt.plot(history.history['loss'])
plt.show()

print('user: ', X_test[:10, 0], 'movie: ', X_test[:10, 1])
print('rate: ', y_test[:10])

print('predict: ', model.predict([X_test[:10, 0], X_test[:10, 1]]))
