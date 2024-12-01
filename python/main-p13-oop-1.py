class Human:
	cry = "A human is crying" # class variable
	happiness = "A human is happy" # class variable
	__slots__ = ('name', 'age') # restrict the attributes that can be added to the class
	def __init__(self,name,age): # initializer
		self.name = name
		self.age = age
	def show(self):
		print(self.name)
		print(self.age)
	@classmethod
	def show_class(cls):
		print(cls.cry)
	@staticmethod
	def show_static():
		print("A human is static")


	# @property	
	# def age(self):
	# 	return self.age
	
	# @age.setter
	# def age(self, value):
	# 	if value < 0:
	# 		raise ValueError("Age can't be negative")
	# 	self.age = value



allen = Human("Allen", 25)
print(allen.happiness)
print(allen)
allen.show()
print(Human.__dict__)
print(isinstance(allen, Human)) # True
print(isinstance(allen, object)) # True
print(isinstance(allen, int)) # False

allen.show_class()

allen.show_static()

# allen.age = 10
# print(allen.age) # 0


class Student(Human):
	def __init__(self, name, age, grade):
		super().__init__(name, age) #super() is used to call MRO (Method Resolution Order) of the parent class
		self.grade = grade

	def show(self):
		super().show()
		print(self.grade)


student = Student("Allen", 25, 10)
student.show()
print(Student.__dict__)
print(student.__class__)
