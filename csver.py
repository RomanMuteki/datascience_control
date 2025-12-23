import os
import pandas as pd

ds_dir = 'data/train'
classes = ['Esox_lucius', 'Cyprinus_carpio', 'Carcharodon_carcharias',
           'Silurus_glanis', 'Delphinapterus_leucas']
image_paths = []
labels = []

for clss in classes:
    class_dir = os.path.join(ds_dir, clss)
    for img_f in os.listdir(class_dir):
        image_paths.append(os.path.join(class_dir, img_f))
        labels.append(clss)

df = pd.DataFrame({'path': image_paths, 'label': labels})
df.to_csv('tree_ds.csv', index=False)