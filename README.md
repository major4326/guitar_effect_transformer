#### Requirements for running the system
The model can only be trained on a NVIDIA GPU. If you have not configured your GPU with pytorch, this guide may be helpful: https://mct-master.github.io/machine-learning/2023/04/25/olivegr-pytorch-gpu.html 

#### Install sox on your computer
To generate the dataset, it is necessary to have sox downloaded on your computer. If your system is linux/ubuntu, this can be done by a single command:
```
sudo apt-get install sox
```

You may refer to this link for other ways to install sox on linux and max: https://arielvb.readthedocs.io/en/latest/docs/commandline/sox.html

##### Installing sox on windows
1. Install sox from this link: https://sourceforge.net/projects/sox/
2. Go to My Computer → Properties → Advanced System Settings → Environment Variables → System variables.
2. Select path
4. Then click Edit → New
5. Add the path to your sox installation. It would likely look something like this: C:\Program Files (x86)\sox-<VERSION NUMBER>\
6. Restart the terminal and type 'sox' to see if the installation has work. It might also be necessary to reboot the system

#### Clone and install requirements
```
pip install -r requirements.txt
```

#### Get the datasets and run the code


