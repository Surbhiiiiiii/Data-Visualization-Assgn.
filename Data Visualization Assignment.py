Data Visualization Assignment
#Functions and methods Homework
1.)
import numpy
def vol(rad):
    pie=numpy.pi
    r=float(input())
    v=(4/3)*pie*r*r*r
    print(v)
vol(2)
#2)
def ran_check(num,low,high):
   if(num<high and num>low):
    print(f'{num} is in range between{low} and {high}')
ran_check(5,2,7)
def ran_bool(num,low,high):
    return num in range(low,high)
ran_bool(3,1,10)
#3)
def up_low(s):
    s.replace(" ","")
    c=0
    d=0
    for char in s:
        if(char.isupper()):
            c=c+1
        elif(char.islower()):
            d=d+1
    print("No. of Upper Case characters:",c)
    print("No. of lower case characters:",d)
s = 'Hello Mr. Rogers, how are you this fine Tuesday?'
up_low(s)
#4)
def unique_list(lst):
    unique=[]
    for num in lst:
        if num not in unique:
            unique.append(num)
    print(unique)
    
unique_list([1,1,1,1,2,2,3,3,3,3,4,5])
#5)
def multiply(numbers):
    x=1
    for num in numbers:
        x=x*num
    print(x)
   
multiply([1,2,3,-4])
#6)
def palindrome(s):
    if(s==s[::-1]):
        print("Palindrome")
    else:
        print("No!Not a plaindrome")
palindrome('helleh')
#7)
import string
def ispangram(str1,alphabet=string.ascii_lowercase):
    str1=str1.lower()
    str1=str1.replace(" ","")
    str1=set(str1)
    alphabet=set(alphabet)
    if(str1==alphabet):
        print("panagram")
    else:
        print("NO")
ispangram("The quick brown fox jumps over the lazy dog")
#Function Practice Excercises
1)
def lesser_of_two_evens(a,b):
        if(a%2==0 and b%2==0):
            if(a<b):
                print(a)
            else:
                print(b)
        else:
            if(a>b):
                print(a)
            else:
                print(b)
lesser_of_two_evens(2,4)
lesser_of_two_evens(2,5)
#2)
def animal_crackers(text):
    s=text.split()
    return (s[0][0]==s[1][0])
animal_crackers('Levelheaded Llama')   
animal_crackers('Crazy Kangaroo')
#3)
def makes_twenty(n1,n2):
    return(n1+n2==20 or n2==20 or n2==20)
makes_twenty(20,10)
makes_twenty(12,8)
makes_twenty(2,3)
#4)
import string
def old_macdonald(name):
        name=list(name)
        name[0]=name[0].upper()
        name[3]=name[3].upper()
        name=''.join(name)
        print(name)
old_macdonald('surbhi')      
#5)
def master_yoda(text):
    x=text.split()
    y=" ".join(x[::-1])
    print(y)
master_yoda('I am home')
master_yoda('We are ready')
#6)
def almost_there(n):
    return(n+10<=100 or n-10>=100 or n+10<=200 or n-10>=200)
almost_there(90)
almost_there(104)
almost_there(150)
almost_there(209)
#7)
def has_33(x):
   for i in range(len(x)-1):
        if x[i:i+2] == [3,3]:
            return True
has_33([1,3,3])
#8)
def paper_doll(text):
    c=[]
    for char in text:
        c.append(char)
        c.append(char)
        c.append(char)
    c="".join(c)   
    print(c)
paper_doll('HEllo')
paper_doll('Mississippi')
#Statement_assessment_test
#1)
st='Print only the words that start with s in  this sentence'
x=st.split()
for x in x:
    if x.startswith('s'):
        print(x)
#2)
for i in range(0,11,2):
    print(i)
#3)
[i for i in range(1,51) if i%3==0]
#4)
st = 'Print every word in this sentence that has an even number of letters'
x=st.split()
for i in x:
    if len(i)%2==0:
        print(i)
#5)
for i in range(100):
    if i%3==0 and i%5==0:
        print("FizzBuzz")
    elif i%3==0:
        print("Fizz")
    elif i%5==0:
        print("Buzz")
    else:
        print(i)
#6)
st = 'Create a list of the first letters of every word in this string'
[i[0] for i in st.split()]