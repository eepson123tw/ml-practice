import csv

with open('test.csv', 'w') as f:
	writer = csv.writer(f)
	writer.writerow(['a', 'b', 'c', 'd'])

with open('test.csv', 'r') as f:
	reader = csv.reader(f, delimiter='|', quotechar="'", skipinitialspace=True)
	for row in reader:
		print(row)
