# Libraries used for machine learning

## Numpy

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


## Pandas
 
 ![](https://cs.uwaterloo.ca/~j2avery/post/2018-01-22-pandas/featured.jpg)
 
 pandas is a python library that provides flexible and expressive data structures (like dataframes and series) for data manipulation. Built on top of numpy, pandas is as fast and yet easier to use.
 
 ![](https://d6vdma9166ldh.cloudfront.net/media/images/1560164130329-Pandas.jpg)
 
 Pandas provides capabilities to read and write data from different sources like CSVs, Excel, SQL Databases, HDFS and many more. It provides functionality to add, update and delete columns, combine or split dataframes/series, handle datetime objects, impute null/missing values, handle time series data, conversion to and from numpy objects and so on. If you are working on a real-world Machine Learning use case, chances are, you would need pandas sooner than later. Similar to numpy, pandas is also an important component of the SciPy or Scientific Python Stack.
 
#### Advantages

- Extremely easy to use and with a small learning curve to handle tabular data.
- Amazing set of utilities to load, transform and write data to multiple formats.
- Compatible with underlying numpy objects and go to choice for most Machine Learning libraries like scikit-learn, etc.
- Capability to prepare plots/visualizations out of the box (utilizes matplotlib to prepare different visualization under the hood).

#### Downsides

- The ease of use comes at the cost of higher memory utilization. Pandas creates far too many additional objects to provide quick access and ease of manipulation.
- Inability to utilize distributed infrastructure. Though pandas can work with formats like HDFS files, it cannot utilize distributed system architecture to improve performance.



 