# open file read file write file

open('file.txt', 'w').write('Hello, world!')

f = open('file.txt', 'r')
print(f.read())
f.close()

with open('file.txt', 'r') as f:
	print(f.read())


class Door:
	def __init__(self):
		self.status = 'closed'
	def __enter__(self):
		print("I am enter")
	def __exit__(self, *args):
		print("I am exit")

with Door() as door:
	print("I am in the with block")
