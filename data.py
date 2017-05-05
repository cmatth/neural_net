
def classOutputs():
	expectedOutput = {		"A" : [1,0,0,0,0,0,0],
							"B" : [0,1,0,0,0,0,0],
							"C" : [0,0,1,0,0,0,0],
							"D" : [0,0,0,1,0,0,0],
							"E" : [0,0,0,0,1,0,0],
							"J" : [0,0,0,0,0,1,0],
							"K" : [0,0,0,0,0,0,1]}
	return expectedOutput

def readFromFile(path):
	with open(path) as f:
		content = f.readlines()

	content = [x.strip('\n') for x in content]

	return content

class datum():
	def __init__(self, label, data):
		self.label = label  #used to identify class of datum
		self.data = data 	#2D array that holds example data

def readIntoClasses(content):
	set = []
	labels = ['A','B','C','D','E','J','K']
	for iteration in range(0,21):
		inst = datum(labels[iteration%7], [])
		for x in range(iteration*9,(9*iteration)+9):
			#inst.data.append(content[x])
			ls = list(content[x])
			for z in ls:
				inst.data.append(z)
		set.append(inst)
		#normalizeData(set)
	return set

def normalizeData(dataSet):
	for x in dataSet:
		for y in range(len(x.data)):
			if x.data[y] == '.':
				x.data[y] = 0
			else:
				x.data[y] = 1







