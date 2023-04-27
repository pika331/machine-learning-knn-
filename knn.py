# the code runs successfully on the kaggle website instead of the local environment. 

import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.neighbors import KNeighborsClassifier

#读取数据
data_train = pd.read_csv(r'/kaggle/input/digit-recognizer/train.csv',engine = 'python',encoding='UTF-8')
data_test = pd.read_csv(r'/kaggle/input/digit-recognizer/test.csv',engine = 'python',encoding='UTF-8')
data_sample = pd.read_csv(r'/kaggle/input/digit-recognizer/sample_submission.csv',engine = 'python',encoding='UTF-8')
print(data_sample.info())

#分割数据
train_np=data_train.values
y=train_np[:,0]
x=train_np[:,1:]

#KNN
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(x, y)

#预测
test_np=data_test.values
ans = knn.predict(test_np)

#输出
id=data_sample.ImageId
result = pd.DataFrame({'ImageId':id, 'Label':ans.astype(np.int32)})
result.to_csv(r"/kaggle/working/submission.csv", index=False)