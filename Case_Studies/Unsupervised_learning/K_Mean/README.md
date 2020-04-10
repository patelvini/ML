# K-Means Clustering

The K-Means clustering algorithm is a classification algorithm that follows the steps outlined below to cluster data points together. It attempts to separate each area of our high dimensional space into sections that represent each class. When we are using it to predict it will simply find what section our point is in and assign it to that class.

### Algorithm:

- **Step 1:** Randomly pick K points to place K **centroids**

    ![](https://miro.medium.com/max/1320/1*G-SF8Y8IAvPJEQaTKIQLSA.png)

- **Step 2:** Assign all of the data points to the centroids by distance. The closest centroid to a point is the one it is assigned to.
     
    Some of the mathematical terms involved in K-means clustering are centroids, euclidian distance. On a quick note centroid of a data is the average or mean of the data and euclidian distance is the distance between two points in the coordinate plane. Given two points A(x1,y1) and B(x2,y2), the euclidian distance between these two points is :

    ![](https://miro.medium.com/max/632/1*T8YEErGM1R7H__pIW32ixw.png)
    
    ![](https://miro.medium.com/max/1318/1*Ke7f0CvFKCFXijCgVNsX0g.png)
    

- **Step 3:** Average all of the points belonging to each centroid to find the middle of those clusters (center of mass). Place the corresponding centroids into that position.

    ![](https://miro.medium.com/max/1322/1*3Jtr8-EfKHztRx1V-wjF9A.png))
    
- **Step 4:** Reassign every point once again to the closest centroid.
- **Step 5:** Repeat steps 3-4 until no point changes which centroid it belongs to.

    ![](https://miro.medium.com/max/1320/1*5ebsmSaH5UfhLLo4QKcxIg.png)
    
    ![](https://miro.medium.com/max/1312/1*NrpSqxWkrQJLEEs7_ZB-aQ.png)
    
    ![](https://miro.medium.com/max/1322/1*vGhNh_8saG9AqLUbKP5s3w.png)
    
    ![](https://miro.medium.com/max/1322/1*FfKg0FDUtxmNkfKw1SaRXw.png)

##### Centroid updates in K-Means

![](https://blog.floydhub.com/content/images/2019/04/kmeans3.gif)

### Flowchart:
![](https://www.researchgate.net/profile/Alhadi_Bustamam/publication/318341309/figure/fig1/AS:514967923159041@1499789328422/Flowchart-of-k-means-clustering-algorithm.png)

## Elbow Method

This is probably the most well-known method for determining the optimal number of clusters.

> Calculate the **Within-Cluster-Sum of Squared Errors (WSS)** for different values of k, and choose the k for which WSS becomes first starts to diminish. In the plot of WSS-versus-k, this is visible as an **elbow**

Within-Cluster-Sum of Squared Errors sounds a bit complex. Let’s break it down:

- The **Squared Error** for each point is the square of the distance of the point from its representation i.e. its predicted cluster center.
- The **WSS score** is the sum of these Squared Errors for all the points.

Any distance metric like the Euclidean Distance or the Manhattan Distance can be used.

![](https://miro.medium.com/max/1664/1*8wV1j-klQA1xFvfaNXuVzg.png)



