# Descriptor: Main file for the python project

class NameValue:
	def __init__(self):
		self._name = 'Aaron'
	def __get__(self, instance, owner): # . is also use __get__ method , self is the NameValue , obj is the instance of the Aaron, owner is the class Cat
		return self._name
	def __set__(self, instance, value):
		if value == 'Aaron':
			self._name = value
			print("Aaron is a cat")
		else:
			print("Aaron is not a cat")
	def __delete__(self, obj):
		print("我被刪掉囉！")
		del self._name

class Cat:
	name = NameValue()
	def __call__(self, *args, **kwds): # __call__ is a magic method that is called when an instance of a class is called as a function
		print("I am called")
		




aaron = Cat()
print(aaron.name) # Aaron
aaron.name = 'Aaron 1'
print(aaron.__dict__)
aaron()
del aaron.name


# type and metaclass , class is an instance of type
# type is a metaclass in python

isinstance(type, type) # True
isinstance(object, type) # True
isinstance(int, type) # True 

print('============= vivi ====================') 
# new is a class method that is called before __init__ method

class Apple:
	def __new__(cls, *args, **kwds):
		print("I am new")
		return super().__new__(cls)
	def __init__(self):
		print("I am init")
	def __contains__(self, item):
		return True
	def __str__(self): # __str__ is called by the str() built-in function and by the print statement
		return "I am a apple"
	def __iter__(self):
		return iter([1, 2, 3])
	def __next__(self):
		return 1

apple = Apple() # I am new , I am init
print('apple' in apple) # True
print(apple)

for i in apple:
	print(i)
