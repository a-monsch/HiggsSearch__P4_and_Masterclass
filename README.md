##### Attention: This branch is undergoing development and will have features that are buggy or not working at all. For a working version --> master branch.

# HiggsSearch Masterclass (pupils and/or students)

## Brief overview
This is a collection of materials that can be used as a template for
tasks that can be worked on in the master class ( or even over several
days) by pupils or by students (the more advanced version). The
sections are factorized and can be adapted and extended according to
the students' level of knowledge. Motivation for this collection is
the discovery of the Higgs Bosons in 2012, therefore the published
[data collections from the CMS detector](http://opendata.cern.ch/record/5500) [[1]](#1) 
from 2012 (8 TeV data set) and the publication of the 
[13 TeV data sets from Atlas](http://opendata.cern.ch/record/15005) [[2]](#2) 
are used.The aim of this introduction is to give the students a 
transparent overview of the discovery of the Higgs boson in the four 
lepton channel and to motivate the statistical reasoning about the 
actual discovery with the help of Python, that is also fundamentally 
presented.

## Language used
 - German
 - English

## Execution viability
It is possible to perform the repo using 
[MyBinder](www.mybinder.org). 
Some of the notebooks rely on graphical applications that require opening in a new window. 
There exists a `_remote_` variant of these, which should be used instead.
For the local application, both options are available (the implementation of the individual 
graphical applications differs slightly due to the different functional scope of the 
packages used, but contains all important features). However, it is recommended to use the 
`_remote` variants, as they fit better into the notebook environment. If this is the first 
time that Python is used, it is recommended to have a look at the notebook containing 
Python basics too.

* To run the repo remotely:   
  [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/a-monsch/HiggsSearch__P4_and_Masterclass/master)

* To run locally (following commands for Terminal/Power Shell):

  ``` 
  git clone https://github.com/a-monsch/HiggsSearch__P4_and_Masterclass
  cd HiggsSearch__P4_and_Masterclass
  ```
  With an optional virtual environment:
  ```
  virtualenv venv_higgs
  # Linux
  source venv_higgs/bin/activate
  # Windows
  .\venv\Scripts\activate
  ```
  The necessary Python (>= 3.6) packages are listed below (`pip install <package>`) but can also be
  downloaded automatically via  
  ```
  pip3 install -r binder/requirements.txt
  ```
   - [SciPy](https://www.scipy.org/)
   - [NumPy](https://numpy.org/)
   - [Pandas](https://pandas.pydata.org/)
   - [matplotlib](https://matplotlib.org/)
   - [kafe2](https://github.com/dsavoiu/kafe2) (current master) (optional)
   - [iminuit](https://iminuit.readthedocs.io/en/latest/)
   - [PyQt5](https://www.riverbankcomputing.com/software/pyqt/intro) (if running locally)
   - [Jupyter](https://jupyter.org/)
   - [tqdm](https://github.com/tqdm/tqdm)
   - [jupyter contrib nbextensions](https://github.com/ipython-contrib/jupyter_contrib_nbextensions) (optional)

  If the virtual environment is used, a kernel for the jupyter notebooks 
  should be reregistered.

  ```
  ipython kernel install --user --name=venv_higgs
  ```

  The jupyteter notebook can be started directly from the `root` 
  directory of the repository with 
  ```
  jupyter notebook
  ```
  After shutting down the notebook, you can leave the virtual environment 
  with `deactivate`.

## Provided datasets
In the MyBinder version the data sets have been downloaded automatically. 
For the local version, the data records can either be downloaded 
[here](https://www.dropbox.com/sh/3j648sojeimjmfh/AACeBAPUZkvsr0gHXULloRSWa?dl=0) manually and unpacked (several times) in a folder `data` or be 
downloaded and unpacked automatically by `sh /binder/postBuild` (Linux). 
Included in the record are  following folders:
- #### for_event_display   
   Contains the `.ig` files for different decays that can be loaded in 
   [IspyWebGL](https://ispy-webgl.web.cern.ch/ispy-webgl/) [[3]](#3) (is called
   in the notebooks). With the help of the values taken there and 
   possible combinations of the detailed tables (also in the folder), 
   the invariant mass of individual decay (four lepton decay) events can 
   be calculated. (Detailed information about the origin of the event files 
   can be found in `README_EVENTS.md` in the folder)
- #### for_widgets
   Contains lists of the Monte Carlo (MC) simulated four lepton invariant 
   masses of the background process and Higgs bosons of different masses 
   (signal process). Is used to display the underlying background and 
   individual signal MCs in the graphical application (in combination with 
   for_event_display calculated masses)
- #### for_longer_analysis
   Contains strongly reduced data sets of the CMS detector from 2012 [[1]](#1) 
   (Run B and C, [more details](http://opendata.cern.ch/record/5500))
   to be processed in the context of the 'for_student' 
   notebook.  Also included is the parameterization of the signal 
   process for different masses using a Gaussian distribution 
   (for the application of the Likelihood Ratio Test). Finally, a 
   summarized list of four lepton invariant masses from the ATLAS 
   measurement, as well as the bin entries of the histograms of the 
   simulated background and signal process are included [[2]](#2)
   ([more details](http://opendata.cern.ch/record/15005)).
   

## References
<a id="1">[1]</a> 
Bin Anuar Afiq Aizuddin Jomhari Nur Zulaiha Geiser Achim. *Higgs-to-four-leptonanalysis example using 2011-2012 data*. 2017. DOI: [10.7483/OPENDATA.CMS.JKB8.D634](10.7483/OPENDATA.CMS.JKB8.D634). URL: [http://opendata.vern/record/5500](http://opendata.vern/record/5500).

<a id="2">[2]</a> 
ATLAS Collaboration. *ATLAS 13 TeV samples collection at least four leptons (electron or muon), for 2020 Open Data release*. URL: [http://opendata.cern.ch/record/15005](http://opendata.cern.ch/record/15005).

<a id="3">[3]</a> 
T McCauley. “A browser-based event display for the CMS Experiment at the LHCusing WebGL”. In:Journal of Physics: Conference Series898 (Okt. 2017), S. 072030. DOI: [10.1088/1742-6596/898/7/072030](10.1088/1742-6596/898/7/072030). URL: [https://doi.org/10.1088%2F1742-6596%2F898%2F7%2F072030](https://doi.org/10.1088%2F1742-6596%2F898%2F7%2F072030).  
