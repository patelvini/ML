# Numpy

![](https://d6vdma9166ldh.cloudfront.net/media/images/1560163943466-NumPy.jpg)

 Numpy is a data handling library, particularly one which allows us to handle large multi-dimensional arrays along with a huge collection of mathematical operations. The following is a quick snippet of numpy in action.
 
 ![](https://d6vdma9166ldh.cloudfront.net/media/images/1560164074833-Numpy.jpg)
 
 Numpy isn’t just a data handling library known for its capability to handle multidimensional data. It is also known for its speed of execution and vectorization capabilities. It provides MATLAB style functionality and hence requires some learning before you can get comfortable. It is also a core dependency for other majorly used libraries like pandas, matplotlib and so on.
 
#### Advantages

Numpy isn’t just a library, it is “the library” when it comes to handling multi-dimensional data. The following are some of the goto features that make it special:

- Matrix (and multi-dimensional array) manipulation capabilities like transpose, reshape,etc.
- Highly efficient data-structures which boost performance and handle garbage collection with a breeze.
- Capability to vectorize operation, again improves performance and parallelization capabilities.

#### Downsides

The major downsides of numpy are:

- Dependency of non-pythonic environmental entities, i.e. due to its dependency upon Cython and other C/C++ libraries setting up numpy can be a pain
- Its high performance comes at a cost. The data types are native to hardware and not python, thus incurring an overhead when numpy objects have to be transformed back to python equivalent ones and vice-versa.