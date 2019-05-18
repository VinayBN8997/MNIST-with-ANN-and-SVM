import csv
from PIL import Image
import numpy as np
import string


csv_File_Path = "A_Z_Handwritten_Data.csv"

count = 1
last_digit_Name =  None

image_Folder_Path = "Alphabets/"



with open(csv_File_Path, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='|')
    count = 0
    for row in reader:
        digit_Name = row.pop(0)
        image_array = np.asarray(row)
        image_array = image_array.reshape(28, 28)
        new_image = Image.fromarray(image_array.astype('uint8'))
        image_Path = image_Folder_Path + str(count) + '.png'
        new_image.save(image_Path)
        count = count + 1

        if count % 1000 == 0:
            print ("Images processed: " + str(count))
