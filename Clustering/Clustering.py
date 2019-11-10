import numpy as np
import matplotlib.pyplot as plt


def distance(X, mu):
  return (((X-mu)**2).sum(axis=1))


def findClosestCentres(X, mu):
  m = X.shape[0]
  (k, n) = mu.shape
  l = list()
  for j in range(k):
    l.append(list())
  for i, x in enumerate(X):
    l[np.argmin(distance(np.full((k, n), x), mu))].append(i)
  return l


def updateCentres(X, C):
  (m, n) = X.shape
  k = len(C)
  mu = np.zeros((k, n))
  for i in range(k):
    mu[i] = X[C[i]].mean(axis=0)
  return mu


def plotData(X, C, mu):
  # plot the data, coloured according to which centre is closest. and also plot the centres themselves
  fig, ax = plt.subplots(figsize=(12, 8))
  ax.scatter(X[C[0], 0], X[C[0], 1], c='c', marker='o')
  ax.scatter(X[C[1], 0], X[C[1], 1], c='b', marker='o')
  ax.scatter(X[C[2], 0], X[C[2], 1], c='g', marker='o')
  # plot centres
  ax.scatter(mu[:, 0], mu[:, 1], c='r', marker='x', s=100, label='centres')
  ax.set_xlabel('x1')
  ax.set_ylabel('x2')
  ax.legend()
  fig.savefig('graph.png')


def main():
  print('testing the distance function ...')
  print(distance(np.array([[1, 2], [3, 4]]), np.array([[1, 2], [1, 2]])))
  print('expected output is [0,8]')

  print('testing the findClosestCentres function ...')
  print(findClosestCentres(
      np.array([[1, 2], [3, 4], [0.9, 1.8]]), np.array([[1, 2], [2.5, 3.5]])))
  print('expected output is [[0,2],[1]]')

  print('testing the updateCentres function ...')
  print(updateCentres(np.array([[1, 2], [3, 4], [0.9, 1.8]]), [[0, 2], [1]]))
  print('expected output is [[0.95,1.9],[3,4]]')

  print('loading test data ...')
  X = np.loadtxt('data.txt')
  [m, n] = X.shape
  iters = 10
  k = 3
  print('initialising centres ...')
  init_points = np.random.choice(m, k, replace=False)
  mu = X[init_points, :]  # initialise centres randomly
  print('running k-means algorithm ...')
  for i in range(iters):
    C = findClosestCentres(X, mu)
    mu = updateCentres(X, C)
  print('plotting output')
  plotData(X, C, mu)


if __name__ == '__main__':
  main()
