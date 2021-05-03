# Data Mining Final Project - README.md file

## GUI 
### Operating the GUI:
Set the Working Directory to the folder "ui" before running the application.

Ensure that you have PyQt5 and its related libraries installed, as well as have your Python application updated to the most recent version.
Check whether the "ui" folder has all the files necessary to run the GUI code.

Within the "ui" folder, run the file called "GUI_BR_PY.py" only to launch the GUI application. You do not need to run "GUI.py" within the Code folder to run the GUI -- it solely contains some code used to connect our GUI with our Python console.

## Inference.py

The steps to produce the clean dataset used in the LPM and Probit (both in Inference.py) are carried out in the EDA.py file, and the dataset itself is exported in line 245 of the EDA file (commented out). For convenience, I have exported the clean dataset and call it directly from the Inference file, precluding the need to run the EDA file as a precursor to the inference file. 
