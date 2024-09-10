# Face Swapping
This project performs face swapping between two images using OpenCV, dlib, and Python. It detects faces in the input images, extracts facial landmarks, and then uses Delaunay triangulation to warp and blend the facial regions from the source image onto the destination image. The result is a face-swapped output where the facial features from one image are seamlessly integrated into another.

### Requirements
To run this project, you'll need to install the required Python packages. You can do this by using the provided requirements.txt file.

### Install 
the dependencies using pip:

```bash
pip install -r requirements.txt
```

### Doenload model
Please download model from from this link  [model fille](https://drive.google.com/file/d/12pjNWnRKK0TYW5dBw_MxYP6vrxnDtTxR/view?usp=drive_link) .  and move it to project root folderr.

###  Run the face swapping program:

```bash
python script.py
```

Make sure to replace "images/one.png" and "images/two.png" with the paths to your source and destination images. The program will save the swapped face images as swapped_face_1.jpg and swapped_face_2.jpg, and display them using OpenCV.

### Files
- face_swap.py: The main Python script for face swapping.
- requirements.txt: A text file containing the necessary Python packages.

> Ensure that the dlib model file shape_predictor_68_face_landmarks.dat is present in the same directory as the script, or provide the correct path to it.
