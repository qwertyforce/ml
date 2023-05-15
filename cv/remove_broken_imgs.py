import os 
from tqdm import tqdm
from PIL import Image
deleted=0
paths= ["./train_generated/watermark/","./train_generated/no_watermark/"]
for p in paths:
    for file_name in tqdm(os.listdir(p)):
        try:
            img = Image.open(p+file_name)
            img.load()
        except:
            os.remove(p+file_name)
            deleted+=1
print(deleted)
