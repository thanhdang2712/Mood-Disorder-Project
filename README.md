# Mood Disorder Package
This is an open-source package developed by research students at Colgate University to study mood disorder-related projects. The package has a Python machine learning repository, which employs some of the most popular feature selection and classification methods to build predictive and analytical models for any dataset, and an R file, which performs statistical analyses.

The package impute missing data, performs grid search to find the best predictive model, and oversample data to get rid of any imbalance in the dataset which makes it easy for people with no coding experience to implement in their own field of research.

## How to use
This document is intended as a detailed guide to help with installation and running of a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

This code has been designed both for use on a local computer or a computer cluster because it runs in parallel. 

- For feature selection and classification, the package is run from main.py where you can enter your file name, specify the dependent variable, and configure the predictive models. The data need not be fully imputed and preprocessed, but the target variable needs to be in binary.
- For statistical analyses, datasets need to be processed into expected forms as input to R functions. Data needs to be fully imputed and processed; the target variable can be treated as continuous or binary depending on the analysis. 

In addition, all the functions in the package are provided with detailed reference for easy navigation.

## Prerequisites

The requirements to run this package are listed in the file requirements.txt.
The package has been run on macOS and cluster. Therefore, the instructions is compatible to macOS and cluster.

### Personal Computer Setup

This package requires the installation of Python software on a 64-bit Windows or a 64-bit MacOS machine or Linux system. You can check your python version
by typing `python3 -V`. If you don't have python 3, please follow instructions on https://realpython.com/installing-python/.
You will also need to have Visual Studio Code and Anaconda installed. 
- https://code.visualstudio.com/download
- https://docs.anaconda.com/anaconda/install/

The following instructions assume that you have never created an anaconda virtual environment on your local computer:
1. Create an environment to run the package on in anaconda. This can be done by typing on the terminal window `conda create -n environmentname python=x.x anaconda`
2. Activate the environment by typing `conda activate environmentname`
3. Install the packages required to run the package. The packages to be installed are already listed in requirements.txt; pip is already installed along with anaconda.n You just need to type `pip install -r requirements.txt` to install all dependencies. If it gives an error, run `./py37/bin/python3 -m pip install --upgrade pip` to update pip and run the previous command again.
4. Type cd and hit enter to move out of the current directory and direct to the main directory
5. Change the current directory (cd) into the location where your copy pf this package is currently located. For example, if the research package is in the folder named final in Desktop, you will need to type `cd Desktop/final`

To run the package on your local computer, type `python3 main.py` to run the package. However, it is recommended that the package is run on cluster to avoid exhausting your PC.

### Cluster Setup
If this is your first time connecting to the Turing Cluster, you can follow the instructions
below to set up your workspace. 

1. To run on cluster, first connect to Colgate VPN (or your Turing hub VPN), after that open terminal and ssh into turing using your turing account.
   Eg. if using Colgate turing account, type: `ssh username@turing.colgate.edu`
2. After that, type in your Turing password.
3. Next, create a new conda environment and activate it like you do when you run on PC. Search Anaconda, go to **Get Additional Installers**, and select “64-Bit (x86) Installer (659 MB)” for both Windows and Mac
4. Install Cyberduck.
- Open Connection: Choose SFTP
- Host: turing.colgate.edu
- Upload the Anaconda Linux file to Cyberduck
5. Then run this in terminal: `bash ~/Anaconda3-2022.05-Linux-x86_64.sh`
6. Install the packages required to run the research package just like when you do it on PC (in the instructions above).

To run on Cluster, cd into the folder and run 'python3 main.py'. Alternatively, you can create a pbs file and run it with these instructions:
1. Create a pbs file in the folder using any text editor. For example, to use nano to create a pbs file, type `nano research.pbs` and hit enter.
2. We need to write on this pbs file in order to run on cluster. A sample of a pbs file is as follows:
> #PBS -l nodes=1:ppn=32
> 
> #PBS -l walltime=100:00:00
> 
> #PBS -e error.txt
> 
> #PBS -o output.txt
> 
> #PBS -m abe
> 
> #PBS -M thdang@colgate.edu
> 
> cd /home/thdang/SPAQ
> 
> /home/thdang/anaconda3
> 
> python3 main.py
-  For the first line, nodes=1 specifies the number of nodes and ppn specifies the number of processors per node.
- The second line specifies the maximum running time 100:00:00 means that the maximum run time is 100 hours.
- The errors will be reported  cha in the file error.txt
- The outputs will be reported in the file output.txt
- Any progress will be reported via email. Replace the email address by your address.
- The next line directs the current directory to the folder you want to run. Modify this line accordingly.
- The next line specifies the anaconda environment you just create. Modify this line according to match the anaconda environment that you just create.
- The last line specifies which python file the pbs file will run on.
- Type Ctrl-X to save what you have type and hit enter to exit
3. Type qsub pbsfilename to submit the job and make it run to the cluster.
4. To check the currently running jobs, we can type qstat -a to see if your job is currently running.


# Contact
Thanh Dang: thdang@colgate.edu
