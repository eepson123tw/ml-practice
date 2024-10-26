import math
print(type(1450))
print(type(1450.0))


#  will return integer //

print(1450//1000)
print(1450/100)

print(3**2)


print(round(3.1415926, 2))


# Banker's Rounding python 3 uses Banker's Rounding

print(round(3.5))
print(round(-4.5))
print(round(0.5))

print(math.pi)
print(math.ceil(3.14))
print(math.floor(3.14))
print(math.sqrt(9))

age = 20
print(float(age))
print(type(str(age)))
print(bool(age))

# scientific notation
print(type(1e3))
print(int(1e3))

nan1 = float('nan')
print(nan1)
print(type(nan1))

# Max 1.7976931348623157e+308


p_inf1 = float('inf')
print(type(p_inf1))  


message = """ 
Hello
Kitty
"""

print(message)


"""
it will lose when it is not assigned to a variable
"""

my_money = 1000000

print(f"I have {my_money} dollars")
print(f"{my_money:,}")  
print(f"{my_money:.2f}") 

ratio = 0.315
print(f"{ratio:.1%}") 


pi = 3.1415926

print(f"|{pi:<20}|")  # 靠左對齊
print(f"|{pi:>20}|")  # 靠右對齊（預設值）
print(f"|{pi:^20}|")  # 置中對齊

print(f"|{pi:x<20.2f}|")


score1, score2 = 123, 1450

print(f"{score1:08}")  # 00000123
print(f"{score2:08}")  # 00001450

hour, minute, second = 3, 12, 7
print(f"{hour:02}:{minute:02}:{second:02}")  # 03:12:07


message = "hellokitty"
print(f"{message[:5]}")

reverse = slice(None, None, -1)
all = slice(None, None, None)
last_5 = slice(-5, None, None)

print(f"{message[reverse]}")
print(f"{message[all]}")
print(f"{message[last_5]}")


data = b"hello"

print(list(data))
