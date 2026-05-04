money = True
if money:
    print("I have money")
    print("I can ride a taxi")

else:
    print("I don't have money")
    print("I need to walk")

a = 10
b = 20
if a > b:
    print("a is greater than b")
else:
    print("a is not greater than b")

cash = 3000
card = True
if cash >= 4000 or card:
    print("I can go to the cafe")
else:
    print("I can't go to the cafe")

l = [1, 2, 3, 4, 5]
if 4 in l:
    print("4 is in the list")
else:
    print("4 is not in the list")

pocket = ["paper", "cellphone", "money"]
if "money" not in pocket:
    pass
else:
    print("I don't have money")

if "wallet" in pocket:
    print("I have a wallet")
elif "coin" in pocket:
    print("I have a coin")
else:
    print("I have nothing")