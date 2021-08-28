"""
==================================
Read a Dataset and plot Pixel Data
==================================

This example illustrates how to open a DICOM file, print some dataset
information, and show it using matplotlib.

"""

# authors : Guillaume Lemaitre <g.lemaitre58@gmail.com>
# license : MIT

import matplotlib.pyplot as plt
from pydicom import dcmread
#from pydicom.data import get_testdata_file

#fpath = get_testdata_file('CT_small.dcm')
ds = dcmread('CT_small.dcm')

# Normal mode:
print()
#print(f"File path........: {fpath}")
print("SOP Class........: {ds.SOPClassUID} ({ds.SOPClassUID.name})")
print()

pat_name = ds.PatientName
display_name = pat_name.family_name + ", " + pat_name.given_name
print("Patient's Name...: {display_name}")
print("Patient ID.......: {ds.PatientID}")
print("Modality.........: {ds.Modality}")
print("Study Date.......: {ds.StudyDate}")
print("Image size.......: {ds.Rows} x {ds.Columns}")
print("Pixel Spacing....: {ds.PixelSpacing}")

# use .get() if not sure the item exists, and want a default value if missing
print("Slice location...: {ds.get('SliceLocation', '(missing)')}")

# plot the image using matplotlib
plt.imshow(ds.pixel_array, cmap=plt.cm.gray)
plt.show()
