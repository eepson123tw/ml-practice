x,y,z= (1,2,3)
x,_,z= (1,2,3)
print(x,z)

# _ will be Anonymous Variable

import keyword 
keyword.kwlist
print(len(keyword.kwlist))

allen = "allen"
george = "george"

allen,george = george,allen
print(allen,george)

age:int = 10


def add(x:int,y:int)->int:
	return x+y

print(add(1,2),'<====add=====>')


## https://pythonbook.cc/chapters/basic/variable
