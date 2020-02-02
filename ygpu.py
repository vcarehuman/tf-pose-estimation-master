##import requests
###import numpy as np
##url_path = 'https://www.indiabix.com'
##r = requests.get(url_path)
##
##from bs4 import BeautifulSoup
##soup = BeautifulSoup(r.text, 'html.parser')
##
##results = soup.select('.div-home-module a')
##del results[58:61]
##
###arr = []
##
##for i in results:
##    concatenate = url_path + i['href']
##    concatenate = concatenate[:-1]
###    print(concatenate)
##    r2 = requests.get(concatenate)
##    soup = BeautifulSoup(r2.text, 'html.parser')
##    results2 = soup.select('a')
##    for a in results2:
##        concatenate2 = concatenate + a['href']
##        r3 = requests.get(concatenate2)
#        
#
#import requests
##import numpy as np
#url_path = 'https://www.indiabix.com/aptitude/probability/'
#r = requests.get(url_path)
#
#from bs4 import BeautifulSoup
#soup = BeautifulSoup(r.text, 'html.parser')
#
#result = soup.find('div',attrs={'class':'bix-div-container'})
#print(result)


import requests
import json
import xlwt
book = xlwt.Workbook()
sheet1 = book.add_sheet("Sheet1") 
num = 1


i = 1
for a in range (i, 400000):
        
    payload = {'keyword' : a}
    
    results = requests.post('https://etea.online/medica_2018_result/action.php?action=search', payload)
    print(results.json())
    t = json.loads(results.content)
    
    for mem in (t['members']):
#        print(mem['name'], "\t"+mem['marks'])
        try:
            row = sheet1.row(num)
            row.write(0, mem['rollno'])
            row.write(1, mem['name'])
            row.write(2, mem['f_name'])
            row.write(3, mem['marks'])
            row.write(4, mem['per_age'])
            num+=1
            book.save("ETEA_result.xls")
        except:
            print("done")
##
#            
##            for step in leg['steps']:
##                print step['html_instructions']
#        
#        
        
        
        
        
        
        
        