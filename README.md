## 2023 Marine Seismology Workshop - OBS data processing


**Instructors**: [Pascal Audet](https://www.uogeophysics.com/authors/admin/)

**When**: Wednesday/Thursday, May 23/24, 2023 at 8:30 AM (ADT). 

**Where**: Room LSC2012 in Life Sciences Centre, Dalhousie University

**What**: This tutorial will provide hands-on experience to process broadband ocean-bottom seismic (BBOBS) data using the Automatic Tilt and Compliance Removal (ATaCR, pronounced "attacker") software. This software is designed to automate, as best as possible, the process of characterizing and removing tilt and compliance noise from vertical component BBOBS data. The tutorial will use the Python implementation of ATaCR. 

---

### Installing the Python version

The software has been pre-installed on the computers in Room LSC2012, therefore there is no need to follow these steps. The following steps provide instructions to install the software on your personal computer.

ATaCR is implemented as a separate module in the open-source Python package OBStools:

- Git repository: [OBStools](https://github.com/nfsi-canada/OBStools)

- Documentation can be found [here](https://nfsi-canada.github.io/OBStools/)

To install `obstools`, we strongly recommend installing and creating a `conda` environment (either from the [Anaconda](https://anaconda.org) distribution or [mini-conda](https://docs.conda.io/en/latest/miniconda.html)) where the code can be installed alongside its dependencies. This **significantly reduces** the potential conflicts in package versions. In a bash (or zsh) terminal, follow these steps:

- Create a conda environment (here we call it `mss` for the name of the symposium) and install `python=3.8` and `obspy`:

```bash
conda create -n msw python=3.8 obspy -c conda-forge
```

- Activate the environment:

```bash
conda activate msw
```

- Install the required [`stdb`](https://github.com/schaefferaj/StDb) package using `pip`:

```bash
pip install stdb
```

Now you're ready to install `obstools`. You might consider one of two options: 1) you want to look at the source code and are considering contributing (awesome!!); 2) you are only interested in using the software and are not interested in the source code.

##### 1) Developer mode: Installing from source

- Navigate on the command line to a path where the software will be installed

- Clone the OBStools repository ([fork](https://docs.github.com/en/github/getting-started-with-github/fork-a-repo) it first, if you are serious about contributing):

```bash
git clone https://github.com/paudetseis/OBStools.git
cd OBStools
```

- Install using `pip`:

```bash
pip install -e .
```

##### 2) User mode: Installing from the Python Package Index (PyPI):

```bash
pip install obstools
```

### Getting the demo data

Finally, download the demo data provided on this github repository by navigating to some work folder (where the data and results of the processing will be located) and typing:

```bash
git clone https://github.com/nfsi-canada/MSW2023.git
cd MSW2023
```

The `DATA` and `EVENTS` folders should now be on your computer and you are ready to start the tutorial.

### Testing your installation

If you want to make sure everything is installed properly, make sure your conda environment has been activated and open a python window by typing in a terminal:

```bash
python
```

which will produce something like:

```bash
Python 3.8.16 (default, Feb  1 2023, 16:05:36) 
[Clang 14.0.6 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> 
```

Then type:

```bash
>>> import stdb
>>> import obstools
```

If nothing happens, you're good to go! If you run into a problem, let us know by [raising an issue](https://github.com/nfsi-canada/MSW2023/issues). 

