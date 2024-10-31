# dict
hero = dict.fromkeys(['name', 'age', 'power', 'health'], None)
hero['name'] = 'Superman'
hero['age'] = 30
hero['power'] = 'Flying'
hero['health'] = 100
print(hero)

print(hero.get('name'))

del hero['age']
print(hero)

pop = hero.pop('power')
print(pop)

popitem = hero.popitem()
print(popitem)

info = {**hero}
print(info)

infotwo = info | hero
print(infotwo)

print(hero.items())


from sys import getsizeof

# 做一個一百萬筆資料的字典
big_dict = {i: i for i in range(1000000)}

keys1 = big_dict.keys()
keys2 = list(big_dict.keys())

print(getsizeof(keys1))  # 40
print(getsizeof(keys2))  # 8000056
