# List 
import copy
hershey = ['chocolate', 'caramel', 'nougat', 'almonds', 'toffee']

if "carame" in hershey:
	print("Caramel is in the list")
else:
	print("Caramel is not in the list")


print(hershey.index('caramel'))
print(hershey.count('caramel'))

hershey.append('peanuts')
hershey.extend(['raisins', 'coconut'])
print(hershey)
hershey.remove('raisins')


sorted_hershey = sorted(hershey, key=len)
print(sorted_hershey)

hershey.clear()
print(hershey)


print(any([False, False, True]))
print(all([False, False, True]))


copy_hershey = copy.deepcopy(sorted_hershey)

# Slice
print(sorted_hershey[1:3])

sorted_hershey[1:3] = ['caramel', 'nougat']
print(sorted_hershey)

sorted_hershey[::2] = [1, 2, 3, 4] # step 3 are replaced with 1, 2, 3, 4
print(sorted_hershey)


# List Comprehension
numbers = [num for num in range(5)]
print(numbers)

zerolist = [ 0 for _ in range(5)]
print(zerolist)

# * operator upacks the list
print([1, 2] * 3)

comics = ['Spider', 'Bat', 'Super']
marvel = ['Iron', 'Thor', 'Hulk']
print(comics + marvel)
all_comics = [*comics, *marvel]
print(all_comics)



# Set
empty_lists = [[] for _ in range(3)]
print(empty_lists)

print([num for num in range(10) if num % 2 == 0])





# Tuple
print('========Tuple========')
# Tuple is immutable
# Tuple is defined by ()
# Tuple is ordered
# Tuple can have duplicate values
# Tuple can have different data types
# Tuple can have nested tuples
# Tuple can have nested lists
# Tuple can have nested dictionaries


data = 1,2
print(data)


# list is not array
# list is a collection of data
# list is ordered
# if want to use array, use from array import array
