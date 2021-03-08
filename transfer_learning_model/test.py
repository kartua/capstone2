#venv = tf
#conda activate tf

import os
import pandas as pd


image_name = []
results = []
for i in os.listdir('test_fire/'):
  command = 'python -m scripts.label_image --graph=tf_files/retrained_graph.pb  --image=test_fire/' + i
  comm = os.popen(command).read()
  object = comm[36:100]
  image_name.append(i)
  results.append(object)
  print(i)

df = pd.DataFrame({'Image': image_name, 'Results': results})
df.to_csv('data1_fire_results.csv')