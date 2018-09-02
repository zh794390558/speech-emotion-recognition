import numpy as np

def dense_to_one_hot(labels_dense, num_classes):
  """Convert class labels from scalars to one-hot vectors.
  Return:
    labels: ndarray of shape [batch_size, num_classes]
  """
  num_labels = labels_dense.shape[0]
  index_offset = np.arange(num_labels) * num_classes
  labels_one_hot = np.zeros((num_labels, num_classes))
  labels_one_hot.flat[ index_offset + labels_dense.ravel()] = 1
  return labels_one_hot


if __name__ == '__main__':
    a = np.random.randint(0, 4, size=5)
    print(a)
    print(dense_to_one_hot(a, 4))


