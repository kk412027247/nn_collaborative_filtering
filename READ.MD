```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

```


```shell script 
    user_id  item_id  rating  timestamp
 0        1     1193       5  978300760
 1        1      661       3  978302109
 2        1      914       3  978301968
 3        1     3408       4  978300275
 4        1     2355       5  978824291
```

```shell script 
   item_id                              titles                        genres
0        1                    Toy Story (1995)   Animation|Children's|Comedy
1        2                      Jumanji (1995)  Adventure|Children's|Fantasy
2        3             Grumpier Old Men (1995)                Comedy|Romance
3        4            Waiting to Exhale (1995)                  Comedy|Drama
4        5  Father of the Bride Part II (1995)                        Comedy
```

```shell script 
movieId  1     50    110   260   296   318   356   480   527   589   593   1196  2571  2858  2959
userId                                                                                           
68        2.5   3.0   2.5   5.0   2.0   3.0   3.5   3.5   4.0   3.5   3.5   5.0   4.5   5.0   2.5
182       4.0   4.5   3.5   3.5   5.0   4.5   5.0   3.5   4.0   2.0   4.5   3.0   5.0   5.0   5.0
249       4.0   4.0   5.0   5.0   4.0   4.5   4.5   4.0   4.5   4.0   4.0   5.0   5.0   4.5   5.0
274       4.0   4.0   4.5   3.0   5.0   4.5   4.5   3.5   4.0   4.5   4.0   4.5   4.0   5.0   5.0
288       4.5   NaN   5.0   5.0   5.0   5.0   5.0   2.0   5.0   4.0   5.0   4.5   3.0   NaN   3.5
307       4.0   4.5   3.5   3.5   4.5   4.5   4.0   3.5   4.5   2.5   4.5   3.0   3.5   4.0   4.0
380       5.0   4.0   4.0   5.0   5.0   3.0   5.0   5.0   NaN   5.0   5.0   5.0   4.5   NaN   4.0
387       NaN   4.5   3.5   4.5   5.0   3.5   4.0   3.0   NaN   3.5   4.0   4.5   4.0   4.5   4.5
414       4.0   5.0   5.0   5.0   5.0   5.0   5.0   4.0   4.0   5.0   4.0   5.0   5.0   5.0   5.0
448       5.0   4.0   NaN   5.0   5.0   NaN   3.0   3.0   NaN   3.0   5.0   5.0   2.0   4.0   4.0
474       4.0   4.0   3.0   4.0   4.0   5.0   3.0   4.5   5.0   4.0   4.5   5.0   4.5   3.5   4.0
599       3.0   3.5   3.5   5.0   5.0   4.0   3.5   4.0   NaN   4.5   3.0   5.0   5.0   5.0   5.0
603       4.0   NaN   1.0   4.0   5.0   NaN   3.0   NaN   3.0   NaN   5.0   3.0   5.0   5.0   4.0
606       2.5   4.5   3.5   4.5   5.0   3.5   4.0   2.5   5.0   3.5   4.5   4.5   5.0   4.5   5.0
610       5.0   4.0   4.5   5.0   5.0   3.0   3.0   5.0   3.5   5.0   4.5   5.0   5.0   3.5   5.0
```

```shell script 
6040 3706 1 5
```
