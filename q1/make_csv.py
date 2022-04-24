import pandas as pd
from glob import glob

all_images = glob("data/Stimuli/*/*.jpg")
# print(all_images)
print(len(all_images))
data = []
for img in all_images:
    stimuli = img.split("/")[2]
    data.append([stimuli, img, img.replace("Stimuli", "FIXATIONMAPS")])

df = pd.DataFrame(data, columns=["Stimuli", "Image", "FixationMap"])
df.to_csv("data/data.csv", index=False)

