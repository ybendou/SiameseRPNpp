import pytube 
import pandas as pd


label_path = '../../TrainingDataset/youtube_bounding_box/'
training_label_path = label_path + 'yt_bb_detection_train/youtube_boundingboxes_detection_train.csv'
validation_label_path = label_path + 'yt_bb_detection_validation/youtube_boundingboxes_detection_validation.csv'


columns = ['youtube_id', 'timestamp_ms', 'class_id', 'class_name', 'object_id', 'object_presence', 
            'xmin', 'xmax', 'ymin', 'ymax']
df_train = pd.read_csv(training_label_path, names=columns) 
# df_val = pd.read_csv(validation_label_path, names=columns) 


save_path = 'D:\\datasets\\TargetBoxDatasets\\YoutubeBoundingBoxes\\videos\\boudingboxes_detection\\'
save_train_path = save_path + 'train\\'

from os import listdir
from os.path import isfile, join
onlyfiles = [f[:-4] for f in listdir(save_train_path) if isfile(join(save_train_path, f))]



counter = 0
max_limit = 1000

print(f'Start scrapping youtube videos... max limit is {max_limit}')
for youtube_id in df_train.youtube_id.unique():
    if youtube_id not in onlyfiles:
        url = f'http://youtu.be/{youtube_id}'
        try : 
            youtube = pytube.YouTube(url)
            video = youtube.streams.first()
            video.download(save_train_path, filename=f'{youtube_id}')
        except:
            print(f'Video {youtube_id} Unavailable')
        counter +=1
        if counter%100==0:
            print(f'Counter at : {counter}')
        if counter>max_limit : 
            print(f'Finishing at : {counter}')
            break
    else : 
        print(f'{youtube_id} already existing')
