import numpy as np
import tensorflow as tf
from collections import deque

q = deque()
q.append([1,2,3])
while q:
    d,e,f = q.popleft()
    print(d)
