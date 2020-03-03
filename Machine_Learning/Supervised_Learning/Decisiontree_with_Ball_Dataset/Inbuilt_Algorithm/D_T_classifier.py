# There are multiple types of balls in our data set as cricket ball and 1 ball.
#This type of balls is classified based on its weight and its surface.
# We must design application using machine learning strategy which is used to classify the balls.

#features -> Weight,Surface

# Assume that 

# Rough = 1
# Smooth = 0

# Labels

# Tennis = 1
# Cricket = 2

from sklearn import tree

data = [[35,1], [47,1], [90,0], [48,1], [90,0], [35,1], [92,0], [35,1], [35,1], [35,1], [96,0], [43,1], [110,0], [35,1], [95,0]]

target_label = [1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 2, 1, 2, 1, 2]

model = tree.DecisionTreeClassifier()

model.fit(data,target_label)

result = model.predict([[99,0]])

print(result)

print(type(result))

if result == 1:
	print("Ball is Tennis.")
else:
	print("Ball is Cricket.")