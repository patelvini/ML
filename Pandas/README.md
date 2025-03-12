# Pandas

![](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcSOP9RXta2RM9EAbqRnJ34CccEbc1cyJ0dITbcGMQbk_mhYzTAe)

**Pandas** is hands down one of the best libraries of python. It supports reading and writing excel spreadsheets, CVS's and a whole lot of manipulation. It is more like a mandatory library you need to know if you’re dealing with datasets from excel files and CSV files. i.e for Machine learning and data science.

# Creating Series, Data Frames and Panels in Pandas

## Series

The Series is a One Dimensional array which is Labelled and it is capable of holding array of any type like: Integer, Float, String and Python Objects. Lets first look at the method of creating Series with Pandas.

> pandas.Series(data, index, dtype, copy)

We can use this method for creating a series in Pandas. Lets discuss how the Series  method takes four arguments:

- **data:** It is the array that needs to be passed so as to convert it into a series. This can be Python lists, NumPy Array or a Python Dictionary or Constants.

- **index:** This holds the index values for each element passed in data. If it is not specified, default is `numpy.arange(length_of_data)`.

- **dtype:** It is the datatype of the data passed in the method.

- **copy:** It takes a Boolean value specifying whether or not to copy the data. If not specified, default is `false`.

#### Using ndarray to create a series:

We can create a Pandas Series using a numpy array, for this we just need to pass the numpy array to the `Series()` Method. We can observe in the output below that the series created has index values which are given by default using the `'range(n)'` where `'n'` is the size of the numpy array.
 
 And at the end of the series the Series() Method prints the datatype of the series created, in this case the datatype is `'int32'`. The numpy array can be of any type: int, float, object, character, etc.
 
 ```
#importing pandas and numpy package
imort pandas as pd
import numpy as np
#creating a array with arange()
numpy_arr = array([2, 4, 6, 8, 10, 20])
#creating a pandas series
pandas_series = pd.Series(arr)
print(pandas_series)
```

**Output**
```
0     2
1     4
2     6
3     8
4    10
5    20
dtype: int32
```

#### Using dictionary to create a series:

We can also create a Pandas Series using a dictionary, for this we just need to pass the dictionary in a pandas `Series()` Method. In the output below the series created has keys of the dictionary as the index and object as the value.

```
#importing pandas package
import pandas as pd
#creating a dictionary
dictionary = {1: 'BMW', 2: 'Audi', 3: 'Mercedes', 4: 'Volkswagen'}
#creating a pandas series
pandas_series = pd.Series(dictionary)
print(pandas_series)
```

**Output**
```
1           BMW
2          Audi
3      Mercedes
4    Volkswagen
dtype: object
```

## Data Frame

A Data Frame is a Two Dimensional data structure. In this kind of data structure the data is arranged in a tabular form (Rows and Columns). Lets first look at the method of creating a Data Frame with Pandas.

> pandas.DataFrame(data, index, columns, dtype, copy)

We can use this method to create a DataFrame in Pandas. Now lets discuss about the arguments required fro DataFrame() Method:

- **data:** The data that is needed to be passed to the `DataFrame()` Method can be of any form line ndarray, series, map, dictionary, lists, constants and another DataFrame.

- **index:** This argument holds the index value of each element in the DataFrame. The default index is `np.arange(n)`.

- **columns:** The default values for columns is `np.arange(n)`.

- **dtype:** This is the datatype of the data passed in the method.

- **copy:** It takes a Boolean value to specify whether or not to copy the data. The default value is `false`.

#### Using lists to create a DataFrame

We can create a Pandas DataFrame using lists. In the output below, we can observe that the Pandas has created a DataFrame with six rows (0, 1, 2, 3, 4, 5) and one column (0).

```
import pandas as pd
li = [1, 2, 3, 4, 5, 6]
pandas_dataframe = pd.DataFrame(li)
print(pandas_dataframe)
```

**Output**
```
   0
0  1
1  2
2  3
3  4
4  5
5  6
```

#### Using Dict of ndarrays and lists to create a DataFrame

We can also use a dictionary of list to create a DataFrame. In the code below we have passed a dictionary of list 'information' to the `DataFrame()` Method and the output is a DataFrame of rows (0, 1, 2, 3, 4) and columns(Brand, Founded).

```
import pandas as pd
information = {'Brand': ['Mercedes', 'Audi', 'BMW', 'Volkswagen', 'Chevrolet'], 'Founded': ['1926', '1969', '1916', '1937', '1911']}
pandas_dataframe = pd.DataFrame(information)
print(pandas_dataframe)
```

**Output**
```
        Brand Founded
0    Mercedes    1926
1        Audi    1969
2         BMW    1916
3  Volkswagen    1937
4   Chevrolet    1911
```

## Panel

A pandas Panel is a 3 Dimensional Container of Data. Lets first take a look at the method of creating Panel with Pandas.

> pandas.Panel(data, item, major_axis, minor_axis, dtype, copy)

- **data:** The data can be of any form like ndarray, list, dict, map, DataFrame.

- **item:** axis 0

- **major_axis:** axis 1

- **minor_axis:** axis 2

- **dtype:** The data type of each column

- **copy:** It takes a Boolean value to specify whether or not to copy the data. The default value is `false`.

#### Using 3D ndarray to create a Panel

We can use a 3 Dimensional ndarray to create a Pandas Panel, let us see an example below. We are using a `random` method to create a ndarray and then we will pass this ndarray to the `pd.Panel()` Method.

```
import pandas as pd
import numpy as np
information = np.random.rand(1, 2, 3)
pandas_panel = pd.Panel(information)
print(pandas_panel)
```

**Output**
```
<class 'pandas.core.panel.Panel'>
Dimensions: 1 (items) x 2 (major_axis) x 3 (minor_axis)
Items axis: 0 to 0
Major_axis axis: 0 to 1
Minor_axis axis: 0 to 2
```

But you will get a `"Future Warning"` from the system when you will try to create a Panel. It says that the Panel will be removed from the pandas library and it also suggests an alternative to creating a panel.

```
sys:1: FutureWarning: 
Panel is deprecated and will be removed in a future version.

The recommended way to represent these types of 3-dimensional data are with a MultiIndex on a DataFrame, via the Panel.to_frame() method

Alternatively, you can use the xarray package http://xarray.pydata.org/en/stable/.

Pandas provides a `.to_xarray()` method to help automate this conversion.
```

#### Using dict of DataFrame to create a Panel

We can also use a dict of DataFrames to create a Panel. We can clearly see from the code below **number of Items: 2, number of major_axis: 1, number of minor_axis: 3**.

```
import pandas as pd
import numpy as np
information = {'randomVal1': pd.DataFrame(np.random.rand(1, 3)),
               'randomVal2': pd.DataFrame(np.random.rand(1, 3))}
pandas_panel = pd.Panel(information)
print(pandas_panel)

```

**Output**

```
<class 'pandas.core.panel.Panel'>
Dimensions: 2 (items) x 1 (major_axis) x 3 (minor_axis)
Items axis: randomVal1 to randomVal2
Major_axis axis: 0 to 0
Minor_axis axis: 0 to 2

```
