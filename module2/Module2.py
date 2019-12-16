import os
import xlwt 
from xlwt import Workbook
from get_choices import *
from get_questions import *
from get_answers import *



print('Enter Directory path of papers: ')
path = input()

print('Enter Model Answer file path: ')
modPath = input()
ModFile = open(modPath,'r')
modelAns = []
for c in ModFile:
    modelAns.append(c[0])


print('Enter Excel File name: ')
Excel = input()

DataSet = os.listdir(path)
imagesList = []
for imgName in DataSet:
    appendImage(str(path) + '/' + imgName ,imagesList)



Out = Workbook()
Grades = Out.add_sheet('Grades')
Grades.write(0,0,'Student ID')
for i in range(0,len(modelAns)):
    Grades.write(0,i+1,'Q'+str(i+1))

for ind in range(0,len(imagesList)):
    result = Correct(imagesList[ind],modelAns)
    print(result)
    Grades.write(ind+1,0,(ind+1))
    for q in range(0,len(result)):
        Grades.write(ind+1,q+1,(result[q]))

Out.save(str(Excel) + '.xls')

print('Excel File ' + str(Excel) + '.xls created successfully')
print('Thank You ^^')
print('Press any key to continue...')
temp = input()