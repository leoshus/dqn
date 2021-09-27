import numpy as np

arr = np.arange(1, 6)
arr = arr[np.newaxis, :]

# print(arr)

# *args 无名参数 但是参数顺序需要按照函数定义的顺序传递
# **kwargs 指定参数名称 参数顺序不需要care
kk = "i am kk , loving life,hello"
print(kk.split(',', 2))
# print(kk.split(2, ','))
print(kk.split(maxsplit=2, sep=','))
