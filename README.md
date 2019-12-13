# Group Implementation Project
### Status: Incomplete
### Incomplete tasks:
#### Send the data back to HTML to render:
I couldn't send the data back to be rendered on the page.
#### Deploy to PythonAnywhere:
To fix some problem that I encountered, keras and tensorflow must be imported, but PythonAnywhere only provides 512 MB of storage, which is not enough for installing both keras and tensorflow libaries. 
### What have been modified since my presentation:
#### The project now is able to predict the class accurately:
The reason to that is my input image was a white back ground image with black number on it (that's what I saw in class - week 3), but when printing the image data out, I realise that it was actually white number on black back ground, so I changed the input image base on that. After that, it rarely failed any of my tests.
### To run the project:
- Run "python app.py" on command line
- Open "http://127.0.0.1:8000/predict" on a browser
- Draw a number and the result will be printed out on the command line (predicted class + confidency)
