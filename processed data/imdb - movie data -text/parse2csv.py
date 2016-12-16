import csv
f2 = open('imdb.csv','w')


inptraincomments = csv.reader(open('imdb_All.csv', 'rb'), delimiter=',', quotechar='|')
for row in inptraincomments:
    count=0
    sent = 0
    sentiment = row[0]
    if sentiment=='positive':
        sent = 1
    else:
        sent = 0
    f2.write('|%s|,|%s|\n'%(str(sent),row[1]))
    

f2.close()
