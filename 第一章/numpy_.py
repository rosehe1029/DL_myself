import numpy
import numpy as np

a=np.array([4,5,6])
print(a.dtype)
print(a.shape)
print(a[0])

b=np.array([[4,5,6],
            [1,2,3]])
print(b.shape)
print(b[0,0],b[0,1],b[1,1])

a=np.zeros((3,3),dtype=int)
b=np.ones((4,5),dtype=int)
c=np.identity((4),dtype=int)
d=np.random.randint(1,10,dtype=int,size=(3,2))
print(a)
print(b)
print(c)
print(d)

a=np.array([[1,2,3,4],
            [5,6,7,8],
            [9,10,11,12]])
print(a)
print(a[2,3],a[0,0])

b=a[0:2,1:3]
print(b)

c=a[1:3]
print(c)
print(c[-1][-1])

a=np.array([[1,2],
           [3,4],
           [5,6]])
print(a[[0,1,2],[0,1,0]])

a=np.array([[1,2,3],
            [4,5,6],
            [7,8,9],
            [10,11,12]])
b=np.array([0,2,0,1])
print(a[np.arange(4),b])

a[np.arange(4),b]+=10
print(a)

x=np.array([1,2])
print(x.dtype)

x=np.array([1.0,2.0])
print(x.dtype)

x=np.array([[1,2],
            [3,4]],dtype=np.float64)

y=np.array([[5,6],
            [7,8]],dtype=np.float64)

print(x+y)
print(np.add(x,y))

print(x-y)
print(np.subtract(x,y))
print(x*y)
print(np.multiply(x,y))
print(np.dot(x,y))
print(x/y)
print(np.divide(x,y))
print(np.sqrt(x))
print(x.dot(y))
print(np.dot(x,y))
print(np.sum(x))
print(np.sum(x,axis=0))
print(np.sum(x,axis=1))

print(np.mean(x))
print(np.mean(x,axis=0))
print(np.mean(x,axis=1))

x=x.T
print(x)
print(np.exp(x))

import matplotlib.pyplot as plt

x=np.arange(0,100,0.1)
y=x*x
plt.plot(x,y)
plt.show()

x=np.arange(0,3*np.pi,0.1)
y=np.sin(x)
plt.plot(x,y)
plt.show()
y=np.cos(x)
plt.plot(x,y)
plt.show()

print(np.nan)
print(np.nan==np.nan)
print(np.inf>np.nan)
print(np.nan-np.nan)
print(0.3==3**0.1)
