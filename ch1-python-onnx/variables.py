# number
a = 1
b = 2.5
c = a + b
print(c)

# string
d = "hello"
e = "12.4"
f = d + " number is " + e
g = 'hello"dsde'
tail = "tail"
word = "i have %d %s" %(3, "dogs") # string formatting
name = "Lee"
age = 30
one = f"My name is {name} and I am {age} years old." # f-string
print(f)
print(g)
print(tail[0:2]) # indexing
print(word)
print(one)
print(tail.upper()) # string metho
print(tail.count("a")) # count occurrences of "a"
print(tail.replace("a", "o")) # replace "a" with "o"

# list
h = [1, 2, 3, 4, 5]
i = ["a", "b", "c", "d", "e"]
del h[0] # delete first element
j = [1, "a", 2.5, [1, 2], {"key": "value"}] # list can contain different types
print(h)
print(i)
print(j)
print(len(h))
del j[2:]
print(j)
h.append(6) # add element to end of list
print(h)
h.sort(reverse=True) # sort list in reverse order
print(h)

dic = {"name": "Lee", "age": 30, "city": "Seoul"}
print(dic)
print(dic["name"])

dic["job"] = "Engineer" # add new key-value pair
print(dic)
del dic["age"] # delete key-value pair
print(dic)
print(dic.get("job")) # get value of "job" key, returns None if key does not exist
print(dic.get("age", "Not found")) # get value of "age" key
print(dic.keys()) # get all keys
print(dic.values()) # get all values
print(dic.items()) # get all key-value pairs
print("name" in dic) # check if "name" key existsp
print("age" in dic) # check if "age" key exists
dic.clear() # clear all key-value pairs
print(dic) # clear() returns None