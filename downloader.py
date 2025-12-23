import os
import pandas as pd
import requests
import time

df = pd.read_csv('data/observ_cpt.csv')
st_time = time.time()
for index, row in df.iterrows():
    '''if index < 3000:
        continue'''
    if index % 30 == 0:
        c_time = time.time()
        print(f"Скачано {index} из 3000 за {c_time - st_time + 1} секунд - {index/(c_time - st_time+1)} из/сек")

    url = row['image_url']
    if index % 10 > 0:
        save_dir = f'data/train/{row["scientific_name"]}'
    else:
        save_dir = f'data/test/{row["scientific_name"]}'
    file = f'{index}.jpg'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    full_path = os.path.join(save_dir, file)

    response = requests.get(url, stream=True)
    with open(full_path, 'wb') as f:
        f.write(response.content)