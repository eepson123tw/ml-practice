import copy
from timeit import timeit
import sys
# Tuple

location = (13.4125, 103.866667)
print(type(location))


number = 1,2,3,4,5
print(type(number))

print(number[0])	


mix = location + number
print(mix) # new tuple

hero = ('Superman', 30, 'Flying', 100)
hero2 = ('Superman', 30, 'Flying', 100)
hero3 = ('Batman', 35, 'Martial Arts', 90)
copy_hero = copy.deepcopy(hero)
print(copy_hero)
copy_hero_two = hero[:]
print(copy_hero_two)

print(copy_hero is copy_hero_two, copy_hero == copy_hero_two,hero is copy_hero, hero == copy_hero)
# because the tuple is immutable, so the copy is the same as the original one

print(hero == hero2, hero is hero2,hero2 is hero,'====')
print(id(hero), id(hero2),id(hero3))
# the value is the same, but the memory address is different



t_list = timeit("heroes = ['悟空', '鳴人', '魯夫']", number=10_000_000)
print("List:", t_list)

t_tuple = timeit("heroes = ('悟空', '鳴人', '魯夫')", number=10_000_000)
print("Tuple:", t_tuple)


print(sys.getsizeof(list(range(10))))  
print(sys.getsizeof(tuple(range(10))))  

# Set

set= {11,1,1,1,2}
print(set)

setNaN = {1,2,3,float('nan')}
print(setNaN)

s1 = {9, 5, 2, 7}
a, *b, c = s1
print(a, b, c)


print(set - setNaN)
