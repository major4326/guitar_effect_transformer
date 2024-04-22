# Electric Guitar Effect Transformer
This is the code behind the paper "Electric guitar effect classification using Transformers". This file provides step-by-step instructions on how to deploy this implementation on your local computer. 

#### Requirements for running the system
The model can only be trained on a NVIDIA GPU. If you have not configured your GPU with pytorch, this guide may be helpful: https://mct-master.github.io/machine-learning/2023/04/25/olivegr-pytorch-gpu.html 

## 1. Install sox on your computer
To generate the dataset, it is necessary to have sox downloaded on your computer. If your system is linux/ubuntu, this can be done by a single command:
```
sudo apt-get install sox
```
You may refer to this link for other ways to install sox on linux and mac: https://arielvb.readthedocs.io/en/latest/docs/commandline/sox.html

### Installing sox on windows
a. Install sox from this link: https://sourceforge.net/projects/sox/

b. Go to My Computer → Properties → Advanced System Settings → Environment Variables → System variables.

c. Select path

d. Then click Edit → New

e. Add the path to your sox installation. It would likely look something like this: C:\Program Files (x86)\sox-(VERSION NUMBER)\

f. Restart the terminal and type 'sox' to see if the installation has worked. It might also be necessary to reboot the system

## 2. Clone and install requirements
Open a terminal in the directory you want to store this repository. Then run this:
```
https://github.com/major4326/guitar_effect_transformer.git
```

After that, you have to install all required dependencies in your prefered environment:
```
cd guitar_effect_transformer
pip install -r requirements.txt
```

## 3. Download the datasets (optional) and generate
You have now two option: you can either run the code on a smaller version of the dataset (which is already in the repository), or you can download whole dataset.
By running the code, you first slice the datasets to 5sec clips, each of these are rendered to 221 effect combinations. Note that if you for some reason want to generate the datasets again, you would have to delete the "gen_multiFX" folders that are created in the /datasets/(DATASET-NAME)_rendered/ directories.
### Option 1: Generate small dataset
Firstly, you have to generate the rendered guitarset (training and validation set). This can be done by running the following code:
```
python generate_dataset ---dataset guitarset -size small
```
This will likely only take a couple of minutes. Furthermore, you have to generate the rendered idmt-smt-guitar dataset, which you do by running this code:
```
python generate_dataset ---dataset idmt-smt -size small
```
### Option 2: Generate the whole dataset
The first thing you have to do is to download the datasets.
* guitarset: https://guitarset.weebly.com/
  * NOTE: download the file called "audio_mono-mic.zip"
* idmt-smt-guitar: https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html
  * NOTE: download "IDMT-SMT-GUITAR_V2.zip"


After that, you have to unzip the files in the "/dataset/" directory. Make sure the paths are "dataset/IDMT-SMT-GUITAR_V2/..." and "dataset/audio_mono-mic/....".


After that, run the code for generating the rendered datasets.

GUITARSET:
```
python generate_dataset ---dataset guitarset
```
IDMT-SMT-GUITARSET:
```
python generate_dataset ---dataset idmt-smt
```

Before running the code, make sure you have enough space on your machine, as the whole rendered dataset would take up around 99GB of disk space. Also keep in mind that generating the whole dataset could take up to several hours depending on your machine (likely between 4-8 hours).

## 4. Testing and training the model
If everything above is done correctly, your system is ready to run the code and test the models. 
#### Testing the models
We have provided a test.py script. You can run this script to test inference of the trained models and alternatively, generate plots.

#### Training the model from scratch
If you wish to train a transformer model from scratch, you can run the train_transformer.py script. These are parameters that you would have to take into account:
* dataset: "small" if you have generated the small dataset, "standard" otherwise
* model: Which model to train on ('ast' or 'wav2vec').

This is an example of training the Audio Spectrogram Transformer from scratch on the small dataset:
```
python --dataset small --model ast
```
