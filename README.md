## **Overview**

- There are two version of the code in this repository: A .py file and a .ipynb file.

- Both python scripts define a neural-network policy for the Half-cheetah environment, and train both a behavioral cloning and policy DAgger algorithm. However, the .py file incorporates a video of the policy being applied in the Half-Cheetah environment.


### **.py File**
- Ensure to import the required libraries to execute the code. Libraries include gymnasium[mujoco], torch, and numpy. You can run the script using "$python Imitation_Learning.py" in the terminal.
  
- The output includes the loss for each algorithm as well as the video of the Half-Cheetah environment. To follow the cheetah in the video, press the tab key.
  
- Only use this script if your computer has a good enough processor. If not (like most Macs), the behavioral cloning algorithm may be okay, but the policy DAgger algorithm will not train properly.

### **.ipynb File**
- Use this file if your processor is not fast enough, and connect to the GPU runtime in Google Colab.
  
- The output is only the training results, since the Half-Cheetah environment video cannot be rendered in a notebook.
  
- Ensure that the expert data file is stored in the "Imitation Learning" folder that the notebook file will create for you

- If you are purely curious to see the model performance, look at the .ipynb file and examine outputs.
  

