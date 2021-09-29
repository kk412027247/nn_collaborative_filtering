>**协同过滤**（英语：Collaborative Filtering），简单来说是利用某兴趣相投、拥有共同经验之群体的喜好来推荐用户感兴趣的信息，个人透过合作的机制给予信息相当程度的回应（如评分）并记录下来以达到过滤的目的进而帮助别人筛选信息，回应不一定局限于特别感兴趣的，特别不感兴趣信息的纪录也相当重要。其后成为电子商务当中很重要的一环，即根据某顾客以往的购买行为以及从具有相似购买行为的顾客群的购买行为去推荐这个顾客其“可能喜欢的品项”，也就是借由社群的喜好提供个人化的信息、商品等的推荐服务。

![Collaborative_filtering.gif](https://upload-images.jianshu.io/upload_images/7505289-baa27e7ba1a76d29.gif?imageMogr2/auto-orient/strip)

#### 评分样本
- 本例子用了`ml-latest-small`这个样本。
- 每一行包括了用户、电影、评分、时间 四个标签。
- 我们可以根据用户过往的电影评分，估算其他没看过电影到评分，进而进行电影推荐。

``` shell script 
    userId  movieId  rating  timestamp
0        1        1     4.0  964982703
1        1        3     4.0  964981247
2        1        6     4.0  964982224
3        1       47     5.0  964983815
4        1       50     5.0  964982931
5        1       70     3.0  964982400
6        1      101     5.0  964980868
7        1      110     4.0  964982176
8        1      151     5.0  964984041
9        1      157     5.0  964984100
10       1      163     5.0  964983650
11       1      216     5.0  964981208
12       1      223     3.0  964980985
13       1      231     5.0  964981179
14       1      235     4.0  964980908
.....
```


#### 需要计算的表格( u*i 的矩阵 R)
- 把评分的样本，转换成一个比较直观的矩阵。
- 表格中NaN的值（被隐藏了），就是我们要计算的值。
- 比如，用户288对 电影1、 110、260、296... 分别进行了评分，那么他对电影50的评分是多少

``` shell script 
movieId  1     50    110   260   296   318   ...  589   593   1196  2571  2858  2959
userId                                       ...                                    
68        2.5   3.0   2.5   5.0   2.0   3.0  ...   3.5   3.5   5.0   4.5   5.0   2.5
182       4.0   4.5   3.5   3.5   5.0   4.5  ...   2.0   4.5   3.0   5.0   5.0   5.0
249       4.0   4.0   5.0   5.0   4.0   4.5  ...   4.0   4.0   5.0   5.0   4.5   5.0
274       4.0   4.0   4.5   3.0   5.0   4.5  ...   4.5   4.0   4.5   4.0   5.0   5.0
288       4.5   NaN   5.0   5.0   5.0   5.0  ...   4.0   5.0   4.5   3.0   NaN   3.5
307       4.0   4.5   3.5   3.5   4.5   4.5  ...   2.5   4.5   3.0   3.5   4.0   4.0
380       5.0   4.0   4.0   5.0   5.0   3.0  ...   5.0   5.0   5.0   4.5   NaN   4.0
387       NaN   4.5   3.5   4.5   5.0   3.5  ...   3.5   4.0   4.5   4.0   4.5   4.5
414       4.0   5.0   5.0   5.0   5.0   5.0  ...   5.0   4.0   5.0   5.0   5.0   5.0
448       5.0   4.0   NaN   5.0   5.0   NaN  ...   3.0   5.0   5.0   2.0   4.0   4.0
474       4.0   4.0   3.0   4.0   4.0   5.0  ...   4.0   4.5   5.0   4.5   3.5   4.0
599       3.0   3.5   3.5   5.0   5.0   4.0  ...   4.5   3.0   5.0   5.0   5.0   5.0
603       4.0   NaN   1.0   4.0   5.0   NaN  ...   NaN   5.0   3.0   5.0   5.0   4.0
606       2.5   4.5   3.5   4.5   5.0   3.5  ...   3.5   4.5   4.5   5.0   4.5   5.0
610       5.0   4.0   4.5   5.0   5.0   3.0  ...   5.0   4.5   5.0   5.0   3.5   5.0
```

#### 隐语义模型
- 用户对某一个电影的最终评分，可能会包含演员、语言、题材、风格之类的属性进行评分，也有可能包含一些自己没有意识到的属性。
- 一部电影本身的因素也会包含演员、语言、题材、风格 等多种属性
- 所以我们会假设用户评价电影的时候，会参照用户自己内心的一个`k维`模型对电影进行评分（这个模型每个人不尽相同）。通过这种方式，用户的项目评分可以通过将用户的每个属性的强度相加来近似计算得出，权重由项目表达该属性的程度决定。这些属性有时被称为隐藏或潜在因子。
- 为了将存在的潜在因子转化为评分矩阵，可以执行以下操作：对于一组大小为 u 的用户 U 和大小为 i 的项目 I，选择任意数量 k 的潜在因子并将大矩阵 R 因式分解为两个更小的矩阵 X（“行因子”）和 Y（“列因子”）。矩阵 X 的维度为 `u × k`，并且矩阵 Y 的维度为`k × i`。如图 2 所示。

![recommendation-system-tensorflow-row-and-column-factors.png](https://upload-images.jianshu.io/upload_images/7505289-60741e916a9c6163.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

- 以上过程，就是一个反向的矩阵相乘，例如，以下是 2 * 3 的矩阵与 3 * 2 到矩阵相乘，得到了 2 * 2的矩阵
![Screenshot from 2020-04-12 20-03-32.png](https://upload-images.jianshu.io/upload_images/7505289-1bcdfff31d052728.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

#### 建模和预测
``` python
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
```
![image.png](https://upload-images.jianshu.io/upload_images/7505289-6cca57baeeac2734.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


- 预测结果
``` shell script 
user:  [431 287 598  41  74  50 353 415 437  72] movie:  [7316  412 3217 2248 1210  149 6416  602 4403 5253]
rate:  [4.5 3.  3.  4.  4.  4.  3.5 4.5 0.5 3.5]
predict:  [[2.873125 ] [3.3347952] [2.657516 ] [3.7854564] [3.5548196] [3.168035 ] [3.7590663] [3.3237288] [2.9627528] [3.9343133]]
```

----
参考文章
[矩阵乘法](https://zh.wikipedia.org/wiki/%E7%9F%A9%E9%99%A3%E4%B9%98%E6%B3%95)
[在 TensorFlow 中构建推荐系统：概览](https://cloud.google.com/solutions/machine-learning/recommendation-system-tensorflow-overview)
[Deep Learning With Keras: Recommender Systems](https://www.johnwittenauer.net/deep-learning-with-keras-recommender-systems/)

