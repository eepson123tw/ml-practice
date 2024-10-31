# Loop

numbers = [1,2,3,4,5]

for num in numbers:
	print(num)

for num in range(1,6):
	print(num)

for char in 'hello':
	print(char)

# variable 

for num in range(1,6):
	hey="123123"
	print(num)

print(num)
print(hey)

# list 

heroes = ['batman', 'superman', 'wonder mon']

print(list(enumerate(heroes)))

heroes = ["悟空", "達爾", "蜘蛛人", "蝙蝠俠"]

for i, hero in enumerate(heroes, 1):
    print(f"{i} {hero}")


# break loop and else

for num in range(1,6):
	if num == 3:
		break
	print(num)
else: # will not execute
	print('Loop completed')

for num in range(1,6):
	if num == 3:
		break
	print(num)
if not num != 3:
	print('Loop completed')
