import sys
import os
from random import shuffle
import numpy as np
import matplotlib
"""
Workaround to solve bug using matplotlib. See:
https://stackoverflow.com/questions/49367013/pipenv-install-matplotlib
"""
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

K_DIMENSION = [1, 2, 3, 6, 10, 20, 30]

def plot_accuracies(accuracies):
  """
  Render the line chart. `matplotlib` conflicts with pipenv 😞. Check:
  https://matplotlib.org/faq/osx_framework.html
  """
  accuracies_pca, accuracies_lda = zip(*accuracies)
  plt.plot(K_DIMENSION, list(map(lambda x : x * 100, accuracies_pca)), 'r-o')
  plt.plot(K_DIMENSION, list(map(lambda x : x * 100, accuracies_lda)), 'b-o')
  plt.axis([0, max(K_DIMENSION) + 2, 0, 120])
  plt.legend(['PCA', 'LDA'], loc='lower right')
  plt.xlabel('Numbers of projection vectors d')
  plt.ylabel('Accuracy (%)')
  plt.show()

def calc_accuracy(mix_set, experiments=1, d0=40):
  """
  Closure to keep sample data and necessary params. Returns the mapping function
  to calculate accuracy rate of given K dimension.
  """
  label_set = set(list(map(lambda x : x["label"], mix_set)))

  def doExperiment(k):
    training_set, test_set = partition_data(mix_set, label_set)

    total_accuracy_PCA = 0
    total_accuracy_LDA = 0
    for _ in range(experiments):
      total_accuracy_PCA += get_PCA_accuracy(training_set, test_set, k)
      total_accuracy_LDA += get_LDA_accuracy(training_set, test_set, d0, k)

    return round(total_accuracy_PCA / experiments, 3), round(total_accuracy_LDA / experiments, 3)

  return doExperiment

def get_PCA_accuracy(training_set, test_set, k):
  """
  Calculate the accuracy rate using PCA
  """
  data = np.array(list(map(lambda x: x['data'], training_set)))
  test_data = np.array(list(map(lambda x: x['data'], test_set)))

  pca = PCA(n_components=k)
  pca_operator = pca.fit(data)

  data_reduced = pca_operator.transform(data)
  test_data_reduced = pca_operator.transform(test_data)

  correct_res = 0

  for idx, test_sample in enumerate(test_data_reduced):
    diff = data_reduced - test_sample
    norms = np.linalg.norm(diff, axis=1)
    closest_idx = np.argmin(norms)
    if (training_set[closest_idx]['label'] == test_set[idx]['label']):
      correct_res += 1

  return correct_res / len(test_data_reduced)

def get_LDA_accuracy(training_set, test_set, d0, k):
  """
  Calculate the accuracy rate using LDA
  """
  data = np.array(list(map(lambda x: x['data'], training_set)))
  label = np.array(list(map(lambda x: x['label'], training_set)))
  test_data = np.array(list(map(lambda x: x['data'], test_set)))

  pca0 = PCA(n_components=d0)
  pca0_operator = pca0.fit(data) # data is the training data set, each row is one training image

  data_reduced = pca0_operator.transform(data) # reduced-dim of the data: rows are examples
  test_data_reduced = pca0_operator.transform(test_data)

  # input of lda is the reduced-dim data from pca:
  lda = LDA(n_components=k) # FLD / LDA
  lda_operator = lda.fit(data_reduced, label)

  trained_data_projection = lda_operator.transform(data_reduced)
  predicted_data = lda_operator.transform(test_data_reduced)

  correct_res = 0

  for idx, prediction in enumerate(predicted_data):
    diff = trained_data_projection - prediction
    norms = np.linalg.norm(diff, axis=1)
    closest_idx = np.argmin(norms)
    if (training_set[closest_idx]['label'] == test_set[idx]['label']):
      correct_res += 1

  return correct_res / len(test_set)


def partition_data(data_set, label_set):
  """
  Randomly shuffle the data and split it into training_set (8) and test_set (2).
  """
  test_set_count = dict.fromkeys(label_set, 0)
  training_set, test_set = [], []

  shuffle(data_set)

  for data in data_set:
    if (test_set_count[data["label"]] >= 2):
      training_set.append(data)
    else:
      test_set.append(data)
      test_set_count[data["label"]] += 1

  return training_set, test_set

def read_and_parse_file(folder_path):
  """
  Read and parse all files from the subdirectory under given folder_path.
  """
  images = []
  paths = os.listdir(folder_path)
  path_names = [folder_path + '/' + path_name for path_name in paths if not path_name.startswith('.')]

  for path_name in path_names:
    files = os.listdir(path_name)
    for file_name in files:
      img = cv2.imread(path_name + '/' + file_name, 0)
      img_col = np.array(img, dtype='float64').flatten()
      data = { 'label': path_name, 'data': img_col }
      images.append(data)

  return images

# --- Leave it here as a reference ---
# def read_pgm(pgmf):
#     """Return a raster of integers from a PGM as a list of lists."""
#     assert pgmf.readline() == b'P5\n'
#     (width, height) = [int(i) for i in pgmf.readline().split()]
#     depth = int(pgmf.readline())
#     assert depth <= 255

#     raster = []
#     for y in range(height):
#       for y in range(width):
#         raster.append(np.frombuffer(pgmf.read(1), dtype=np.uint8).item())
#     return raster
# ------------------------------------

if __name__ == '__main__':
    folder_path = sys.argv[1]
    experiments = int(sys.argv[2])
    d0 = int(sys.argv[3])

    # Parse data
    images = read_and_parse_file(folder_path)

    # Get mapped accuracy
    toAccuracy = calc_accuracy(images, experiments=experiments, d0=d0)
    accuracies = list(map(toAccuracy, K_DIMENSION))

    # Plot the accuracy with nice line chart
    plot_accuracies(accuracies)