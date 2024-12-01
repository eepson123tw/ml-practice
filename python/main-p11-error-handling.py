# Error Handling

## Raise an exception

class ErrorName(Exception):
	def __init__(self, message):
		self.message = message
		super().__init__(self.message)
	def __str__(self):
		return f"this is a custom => {self.message}"

# raise ErrorName("This is an error message")


## try except

try:
	1/0
except ZeroDivisionError as e:
	print("You can't divide by zero =>",e)


print("This is the end of the program")

print("====================================")

## try except finally

try:
	1/0
except ZeroDivisionError as e:
	print("You can't divide by zero =>",e)
finally:
	print("This is finally block")	
