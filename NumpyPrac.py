import numpy as np
import time
import sys
a = np.array([1,2,3])
Two_d = np.array([(1,2,3),(4,5,6)])
print(a)
print(Two_d)
# comparing the storage used

# print(a[0])
b = range(1000) #list
print("Space used by a list",sys.getsizeof(5)*len(b))#getsizeof(_element_) gives the space occupied by element
#multiplying it with the length of the array we get the memory used/ space occupied
c = np.arange(1000) #array
print("Space used by a Array",c.size*c.itemsize)


# comparing the time taken to find sum using list and then an array
size = 100000
L1 = range(size) # this is a list
L2 = range(size) # this is a list

A1 = np.arange(size)# it is a numpy array
A2 = np.arange(size) #   it is a numpy array

# we have two list and two arrays of same size and we will find sum of both list
# and array to find the time taken
start = time.time() #initializing a variable called time
result = [(x+y) for x,y in zip(L1,L2)]
print("The result obtained is : ",result)
print("Python list book :", (time.time()-start)*1000)

start = time.time()
result = A1 + A2
print("Python array book :", (time.time()-start)*1000)

