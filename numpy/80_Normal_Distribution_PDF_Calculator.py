import math

def normal_pdf(x, mean, std_dev):
	"""
	Calculate the probability density function (PDF) of the normal distribution.
	:param x: The value at which the PDF is evaluated.
	:param mean: The mean (Î¼) of the distribution.
	:param std_dev: The standard deviation (Ï) of the distribution.
	"""
	# Your code here
	val = (1/(std_dev*(math.sqrt(2*math.pi))))*(math.e**(-((x-mean)**2)/(2*(std_dev**2))))
	return round(val,5)