
def classOutputs():
	return {	"A" : [1,0,0,0,0,0,0],
				"B" : [0,1,0,0,0,0,0],
				"C" : [0,0,1,0,0,0,0],
				"D" : [0,0,0,1,0,0,0],
				"E" : [0,0,0,0,1,0,0],
				"J" : [0,0,0,0,0,1,0],
				"K" : [0,0,0,0,0,0,1]}

def labelIndex():
	return {	0 : 'A',
				1 : 'B',
				2 : 'C',
				3 : 'D',
				4 : 'E',
				5 : 'J',
				6 : 'K' }

def readFromFile(path):
	with open(path) as f: content = f.readlines()
	content = [x.strip('\n') for x in content]
	return content

class datum():
	def __init__(self, label, data):
		self.label = label  #used to identify class of datum
		self.data = data 	#2D array that holds example data

def readIntoClasses(content):
	set = []
	labels = ['A','B','C','D','E','J','K']
	length = len(content) / 9
	for iteration in range(0,length):
		inst = datum(labels[iteration%7], [])
		for x in range(iteration*9,(9*iteration)+9):
			ls = list(content[x])
			for z in ls:
				inst.data.append(z)
		set.append(inst)
	return set

def normalizeData(dataSet):
	for x in dataSet:
		for y in range(len(x.data)):
			if x.data[y] == '.':
				x.data[y] =  -1
			else:
				x.data[y] = 1

def getDataSets(training_path, testing_path):
	datas = readFromFile(training_path)
	trainingSet = readIntoClasses(datas)
	datas = readFromFile(testing_path)
	testingSet = readIntoClasses(datas)
	normalizeData(trainingSet)
	normalizeData(testingSet)
	tmp = []
	for x in range(0, 14):
		tmp.append(testingSet[x])
	for x in tmp:
		trainingSet.append(x)
		testingSet.remove(x)
	return trainingSet, testingSet







