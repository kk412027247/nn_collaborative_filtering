import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model
from tensorflow.keras.layers import Activation, Lambda, Embedding, Input, Concatenate, Dense, Dropout, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

douyin = pd.read_csv('data.csv')

douyin = douyin.dropna()
douyin = douyin.astype({'comment_user_uid': 'int64'})

uids = np.array(douyin['comment_user_uid'])
u, c = np.unique(uids, return_counts=True)
dup = u[c >= 5]

douyin = douyin.loc[douyin['comment_user_uid'].isin(dup)]

douyin['user_uid_video_id'] = douyin['comment_user_uid'].astype(str) + douyin['video_id'].astype(str)

douyin['commit'] = 1

douyin = douyin.drop_duplicates(subset='user_uid_video_id')

# print(douyin)

del douyin['user_uid_video_id']

d = {}

for _, row in douyin.iterrows():
    if (row['video_id'] in d):
        d[row['video_id']].append(row['comment_user_uid'])
    else:
        d[row['video_id']] = [row['comment_user_uid']]

# index             video_id comment_text  create_time  comment_user_uid                desc             tag  commit


data_2 = {'index': [], 'video_id': [], 'comment_text': [], 'create_time': [], 'comment_user_uid': [], 'desc': [],
          'tag': [], 'commit': []}

uids_2 = np.unique(np.array(douyin['comment_user_uid']))

for key in d:
    random_user_ids = [item for item in np.random.choice(uids_2, 5) if item not in d[key]]
    data_2['comment_user_uid'].extend(random_user_ids)
    ids_counter = list(range(len(random_user_ids)))
    data_2['video_id'].extend([key for _ in ids_counter])
    data_2['commit'].extend([0 for _ in ids_counter])
    data_2['index'].extend([0 for _ in ids_counter])
    data_2['comment_text'].extend(['-' for _ in ids_counter])
    data_2['create_time'].extend([0 for _ in ids_counter])
    data_2['tag'].extend([() for _ in ids_counter])
    data_2['desc'].extend(['-' for _ in ids_counter])

douyin_2 = pd.DataFrame(data=data_2)

print(douyin_2)

print(douyin.head(1))

douyin = pd.merge(
    douyin,
    douyin_2,
    how='outer',
    on=['index', 'video_id', 'comment_text', 'create_time', 'comment_user_uid', 'desc', 'tag', 'commit']
)

print(douyin)

user_enc = LabelEncoder()
douyin['user'] = user_enc.fit_transform(douyin['comment_user_uid'].values)
n_users = douyin['user'].nunique()

item_enc = LabelEncoder()
douyin['video'] = item_enc.fit_transform(douyin['video_id'].values)
n_movies = douyin['video'].nunique()

X = douyin[['user', 'video']].values
y = douyin['commit'].values

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
x = Lambda(lambda x: x * (1 - 0) + 0)(x)

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
