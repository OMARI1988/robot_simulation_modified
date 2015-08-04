from multiprocessing import Process
import multiprocessing

def func1():
  print 'func1: starting'
  for i in xrange(1000000000):	  pass
  print 'func1: finishing'

def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()



def calc(t):
  print t
  print 'calc: starting'
  for i in xrange(10000):	  pass
  print 'calc: finishing'
  return (str(t[0]),)


if __name__ == '__main__':
	#runInParallel(func1, func1, func1, func1, func1, func1)
	pool = multiprocessing.Pool(6)
	out1 = zip(*pool.map(calc, [[1,1,1],[1,2,2],[3,3,3]]))
	print out1
