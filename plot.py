import numpy as np
import matplotlib.pyplot as plt

def plotDataScatter(title, dataX, dataY, xLabel, yLabel):
	colors = (0, 0, 0)
	area = np.pi * 3

	# Plot
	plt.scatter(dataX, dataY, s=area, c=colors, alpha=0.5)
	plt.title(title)
	plt.xlabel(xLabel)
	plt.ylabel(yLabel)
	plt.show()


'''
# Create data
N = 50000
x = np.random.randint(0,100,N)
y = np.random.randint(0,100,N)
colors = (0, 0, 0)
area = np.pi * 3

# Plot
plt.scatter(x, y, s=area, c=colors, alpha=0.5)
plt.title('Scatter plot pythonspot.com')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
'''