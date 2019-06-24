import turicreate as tc
import os

"""
- load_images returns SFrame objects 
  which essentially acts as an ordered dict of SArrays:
  each dict is a column, such as "path,image"
  each row in any dict is part of SArray
- Loades images from a directory (jpg)
- Params:(url, format='auto', with_path=True, recursive=True, ignore_failure=True, random_order=False)
  'with_path = True' indicates that path column will be added to the defuault image column
- Printed format for SFrame which is returned: path: (...), image: (...)
"""
data = tc.image_analysis.load_images("dog-breed-dataset-master/")

"""
The function which passed to 'SArray.apply' method will be called for every row.
-  SFrame->SArray["path"]
-  data["<new_column_name"]->creates new dict in SFrame
-  The method which is called upon every row taking the image path and returns its folder name:
   1-> os.path.dirname("C:/folder1/folder2/filename.xml") = 'C:/folder1/folder2'
   2-> os.path.basename('C:/folder1/folder2') = 'folder2'
-  Any folder name is essentially the label of given dog picture
"""
data["label"] = data["path"].apply(lambda path: os.path.basename(os.path.dirname(path)))

# Saves SFrame object for later use
data.save("dog_classifier.sframe")

print("Completed creating SFrame of categorized images to labels")
