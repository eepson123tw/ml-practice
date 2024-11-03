from random import random

def functionCall():
	pass



def numToSting(num:int)->str:
	return str(num)

print(type(numToSting(123))) # str


# Keyword Argument

def greet(name, msg):
	print(f'Hello {name}, {msg}')

greet('World', 'Good morning')
greet(name='World', msg='Good morning')
greet(msg='Good morning', name='World')


# * /

def print_something(a, b, c, *, d, e):
    print(a, b, c, d, e)

print_something(1, 2, 3, d=4, e=5)

print(1, 2, 3, 4, 5, sep="、", end="。\n")

def add_random_value(number, value=None):
    value = value or random()
    print(f"{number=} {value=}")

add_random_value(1)
add_random_value(2)


# all arguments

def print_args(*args):
	print(args)

print_args(1, 2, 3, 4, 5)
print_args()

def print_kwargs(*a,**kwargs):
	print(kwargs,a)


print_kwargs(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15, name='World', msg='Good morning')

 
# combine all & Docstring

def print_all(a,b,c,*args,**kwargs):
	"""
		Combine all
		a,b,c: positional arguments
		args: tuple of positional arguments
		kwargs: dictionary of keyword arguments
	"""
	print(a,b,c,args,kwargs)

heroes = ['batman', 'superman', 'wonder mon']
print_all(*heroes,name='World', msg='Good morning')

print(print_all.__doc__)


# LEGB Rule

def test_LEGB():
	x = 'local x'
	
	def inner():
		nonlocal x
		x = 'nonlocal x'
		print("inner:", x)
	
	inner()
	print("test_LEGB:", x)

test_LEGB()
print(globals())
# print("global:", x)


# keyword global | nonlocal

x = 'global x'

def test_global():
	global x
	x = 'global x 1'
	print("test_global:", x)

test_global()
print("global:", x)


"""Q: What are the rules for local and global variables in Python?

A: In Python, variables that are only referenced inside a function are implicitly global. If a variable is assigned a value anywhere within the function’s body, it’s assumed to be a local unless explicitly declared as global."""


# inspect

import inspect

def test_inspect():
	x = 'local x'
	print(inspect.currentframe().f_locals)

# ismethod or isfunction

def test_isfunction():
	def a():
		pass
	print(inspect.isfunction(a))
	print(inspect.ismethod(a))

test_inspect()
test_isfunction()
