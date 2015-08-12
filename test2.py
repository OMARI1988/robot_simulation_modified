import numpy

a = {}
a[0] = 2
a[1] = 3
a[2] = 4
a[3] = 5
a[4] = 6
print [[subset,[a[i] for i in subset]] for subset in [[0,1],[1,3]]]
