
import csv

lines = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    image_path = line[0]
    image_file_name = image_path.split('\\')[-1]
