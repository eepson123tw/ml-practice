print(float(True))
print(float(False))
print(1==True)
print(0==False)
print(1.0==True)
message = "Hello, World!"
print(message[True])

# Logical Operators
print('========Logical Operators========')
print('True and True ====>',True and True)
print('True and False =====>',True and False)
print('False and True =====>',False and True)
print('False or False =====>',False or False)
print('True or False =====>',True or False)
print('not True =====>',not True)
# Short-circuit evaluation
print('Short-circuit evaluation')
False and print('This will not print')
True or print('This will not print')
True and print('This will print')

# Flow Control
print('========Flow Control========')
# if statement
print('if statement')
x = 10
if x > 5:
	print('x is greater than 5')
if x < 20:
	print('x is less than 20')
if x == 10:
	print('x is 10')
# if-else statement
print('if-else statement')
if x > 20:
	print('x is greater than 20')
else:
	print('x is less than 20')
# if-elif-else statement
print('if-elif-else statement')
if x > 20:
	print('x is greater than 20')
elif x < 20:
	print('x is less than 20')
else:
	print('x is 20')

# Terenary Operator
print('========Terenary Operator========')
x = 10
y = 20
max = x if x > y else y
print(max)

# match statement
print('========match statement========')
x = 10
match x:
	case 1:
		print('x is 1')
	case 2:
		print('x is 2')
	case 3:
		print('x is 3')
	case _:
		print('x is not 1, 2 or 3')

data = 100
match data:
	case int() | float():
		print('data is a number')
	case str():
		print('data is a string')
	case _:
		print('other data type')

user_data = ['John', 30]
match user_data:
	case ['John', age]:
		print(f'John is {age} years old')
	case [name, age]:
		print(f'{name} is {age} years old')
	case _:
		print('Invalid data')

mendict = {'name':'John', 'age':30}
match mendict:
	case {'name':'John', 'age':30}:
		print('John is 30 years old')
	case {'name':name, 'age':age}:
		print(f'{name} is {age} years old')
	case _:
		print('Invalid data')

number = [2,3]
match number:
	case x ,y if x % 2 ==0:
		print(f'{x} is even')
	case _:
		print('Odd number')

numbers = [0,1,2,3,4,5]
match numbers:
	case [0,1,2,*rest]:
		print('First three numbers are 0, 1 and 2')
		print('Other numbers are:', rest)
	case _:
		print('Invalid data')


def fib(n):
	match n:
		case 0:
			return 0
		case 1:
			return 1
		case _:
			return fib(n-1) + fib(n-2)

print(fib(10))


# id
print('========id========')
x = 10
y = 10
data = [1,2,3]
print(id(x))
print(id(y))
print(id(data))
