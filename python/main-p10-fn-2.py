# lambda function
"""
A lambda function is a small, anonymous function that can have any number of arguments but can only have one expression. They are also known as "anonymous functions" because they don't have a name like regular functions defined with def.
"""
add = lambda *args: sum(args)
print(add(1, 2, 3, 4, 5)) 


print(sorted([1, 2, 3, 4, 5], key=lambda x: x % 2 != 0))
print(list(filter(lambda x: x % 2 == 0, [1, 2, 3, 4, 5])))


# closure
"""
A closure is a function object that has access to variables in its enclosing scope even if the enclosing function is no longer in the memory.
"""

def outer_function(msg):
	def inner_function():
		print(msg)
	
	return inner_function

hi_func = outer_function('Hi')
hello_func = outer_function('Hello')

hi_func()
hello_func()

print(hi_func.__closure__[0].cell_contents)
print(hi_func.__code__.co_freevars)
print(hi_func.__code__.co_cellvars)


# decorator
"""
A decorator is a design pattern in Python that allows a user to add new functionality to an existing object without modifying its structure. Decorators are usually called before the definition of a function you want to decorate.
"""

def decorator_function(original_function):
	def wrapper_function(*args, **kwargs):
		print(f'wrapper executed this before {original_function.__name__}')
		return original_function(*args, **kwargs)
	
	return wrapper_function

def display(msg):
	print('display function ran',msg)

decorated_display = decorator_function(display)
decorated_display(msg='Hello')

@decorator_function
def display_info(name, age):
	print(f'display_info ran with arguments {name} and {age}')

display_info('John', 25)


print(display_info)

# Recursion

def factorial(n):
	if n == 0:
		return 1
	return n * factorial(n - 1)

print(factorial(10))

# yield

def generator_function(num):
	for i in range(num):
		yield i

g = generator_function(10)
print([g for g in g])
