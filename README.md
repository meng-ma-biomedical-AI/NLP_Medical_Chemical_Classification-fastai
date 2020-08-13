
# Relevant Medical Chemical Classification via NLP using FastAI

### Project Overview
This project aims to classify relevant chemicals from medical texts via Natural Language Processing using the FastAI library. The goal was to combine NLP, the medical domain, and automated machine learning pipelines using the Igauzio Data Platform.

The first step was to train a language model via transfer learning from a Wikipedia model to obtain language encodings that include the medical terminology. Using these language encodings, the next step was to train a classification model for the relevant chemicals per abstract.

The project is in the form of a Kubeflow Pipeline using MLRun and Nuclio on the Iguazio Data Platform. The pipeline includes:

 1. Loading and sampling a subset of the data
 2. Creating DataBunches for the language and classification model
 3. Hyper-parameter tuning for the language model
 4. Training of the language model
 5. Hyper-parameter tuning for the classification model
 6. Training of the classification model
 7. Deploying a model server for realtime inference
 8. Deploying a model server tester to do sample inferencing

### Data Overview
The data for this project was mined from PubMed. Their website states: "PubMed® comprises more than 30 million citations for biomedical literature from MEDLINE, life science journals, and online books. Citations may include links to full-text content from PubMed Central and publisher web sites."

This project uses a free sample of data with over 3,000,000 articles that can be found here: [ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline-2018-sample/](ftp://ftp.ncbi.nlm.nih.gov/pubmed/baseline-2018-sample/).

### Downloading Data
The file `data_cleaning/links.txt` was compiled using this sample. From there, `data_cleaning/get_data.py` was used to download all the files via FTP using wget.
```python
import subprocess

with open("links.txt", 'r') as f:
	for line in f:
		print(line.strip())
		subprocess.run(f'wget -P data/ {line.strip()}', shell=True)
```

### Processing Data
The data itself is comprised of many XML files. Within these XML files, there is quite a bit of data on each of the articles including publication dates, titles, authors, abstracts, relevant chemicals, journals, publishers, and more.

Although this project was mainly concerned in the abstract and relevant chemicals, the mined dataset also includes information on the titles, authors, dates published, and publication journals. The data mining was done using the `data_cleaning/DataCleaning.ipynb` notebook.

```python
import xml.etree.cElementTree as ET
from datetime import datetime
import pandas as pd
from glob import glob
import numpy as np
import os

data_files = sorted(glob("data/*.xml"))
for i, file in enumerate(data_files):
    print(f"Processing {i+1} out of {len(data_files)}")
    
    # Check for existing pickle
    if os.path.exists(f"pickle/frame_{i+1}.pkl"):
        continue
    
    tree = ET.parse(file)

    root = tree.getroot() 

    articles = root.findall("PubmedArticle")

    data = []
    for article in articles:
        # Article Title
        try:
            title = article.find("MedlineCitation/Article/ArticleTitle").text
        except:
            title = np.NaN

        # Original Publish Date
        try:
            date_orig = datetime(int(article.find("MedlineCitation/DateCompleted/Year").text),
                                 int(article.find("MedlineCitation/DateCompleted/Month").text),
                                 int(article.find("MedlineCitation/DateCompleted/Day").text))
        except:
            date_orig = np.NaN

        # Revised Publish Date
        try:
            date_revised = datetime(int(article.find("MedlineCitation/DateRevised/Year").text),
                                 int(article.find("MedlineCitation/DateRevised/Month").text),
                                 int(article.find("MedlineCitation/DateRevised/Day").text))
        except:
            date_revised = np.NaN

        # Journal Title
        try:
            journal = article.find("MedlineCitation/Article/Journal/Title").text
        except:
            journal = np.NaN
        
        # Authors
        try:
            authors = [i.find("LastName").text + ", " + i.find("ForeName").text for i in article.findall("MedlineCitation/Article/AuthorList/Author")]
        except:
            authors = np.NaN

        # Abstract
        try:
            abstract = [i.text for i in article.find("MedlineCitation/Article/Abstract")][0]
        except:
            abstract = np.NaN

        # List of Chemicals
        try:
            chemicals = [i.text for i in article.findall("MedlineCitation/ChemicalList/Chemical/NameOfSubstance")]
            
            if len(chemicals) == 0:
                chemicals = np.NaN
        except:
            chemicals = np.NaN

        data.append({'title': title,
                     'date_orig': date_orig,
                     'date_revised':date_revised,
                     'journal': journal,
                     'authors' : authors,
                     'abstract' : abstract,
                     'chemicals' : chemicals})

    df = pd.DataFrame(data)
    df.to_pickle(f"pickle/frame_{i+1}.pkl")
```

This took a substantial amount of time to run to completion. In order to prevent having to do this again, the data was saved as a master dataframe pickle.
```python
from datetime import datetime
import pandas as pd
from glob import glob
import numpy as np
import os

pickles = sorted(glob("pickle/*.pkl"))
dataframes = []
for i, file in enumerate(pickles):
    print(f"Processing {i+1} out of {len(pickles)}")
    df = pd.read_pickle(file)
    dataframes.append(df)
master_df = pd.concat(dataframes)
master_df.to_pickle(f"pickle/master_frame.pkl")
```

Finally, there was quite a bit of missing data. Since transfer learning doesn't need nearly as much data to train a model as it would to train one from scratch, all the missing rows in the master dataframe was dropped and several sample datasets were made.

```python
df = pd.read_pickle("pickle/master_frame.pkl")
df_trimmed = df[["title", 'abstract', 'chemicals']].dropna()
df_trimmed.to_pickle("pickle/master_frame_trimmed.pkl")

df_trimmed = pd.read_pickle("pickle/master_frame_trimmed.pkl")
df_sample_10000 = df_trimmed.sample(n=10000, random_state=1)
df_sample_10000.to_pickle("pickle/master_frame_sample_10000.pkl")

df_trimmed = pd.read_pickle("pickle/master_frame_trimmed.pkl")
df_sample_20000 = df_trimmed.sample(n=20000, random_state=1)
df_sample_20000.to_pickle("pickle/master_frame_sample_20000.pkl")

df_trimmed = pd.read_pickle("pickle/master_frame_trimmed.pkl")
df_sample_50000 = df_trimmed.sample(n=50000, random_state=1)
df_sample_50000.to_pickle("pickle/master_frame_sample_50000.pkl")

df_trimmed = pd.read_pickle("pickle/master_frame_trimmed.pkl")
df_sample_100000 = df_trimmed.sample(n=100000, random_state=1)
df_sample_100000.to_pickle("pickle/master_frame_sample_100000.pkl")
```

If you would like to use this project but don't want to download and pre-process all of the data, the `master_frame.pkl` (27 GB) and `master_frame_trimmed.pkl`  (13 GB) files will be made available via download: LINK.

### Initial Model Development
The initial model development and experimentation was done in the notebook `ModelDev.ipynb`. It is the basis for which the different pipeline components were based upon. The model pipeline was initially developed in this single notebook environment and was then later separated into different components to work with MLRun on the Iguazio Data Platform. The notebook itself is quite messy as there is model code, testing code, and troubleshooting code all in one place. It has been included for the sake of completion.

### MLRun Pipeline Components
Once everything was working in the initial notebook, the different components were separated into their own notebooks. MLRun is based on a Jupyter Notebook environment where each pipeline component has its own notebook. From there, a Kubernetes resource in the form of a  `yaml` file is exported for use within the pipeline code itself.

The source code for the components can be found within the `components/notebooks` directory. The outputted Kubernetes resources can be found within the `components/yaml` directory.

### MLRun Image Creation
One important aspect of any pipeline is the environment in which each component will execute. In a Kubeflow pipeline, separate docker containers are used for each pipeline component. Iguazio allows the user to easily create new images and push to the internal Docker repository for use in pipelines.

This was done using the notebook `components/notebooks/CreateImages.ipynb`:
```python
from mlrun import mlconf, NewTask, mount_v3io,new_function, code_to_function

docker_registry = 'docker-registry.default-tenant.app.groupwaretech.iguazio-c0.com:80'
new_image = '/fastai'

image_fn = new_function(kind='job')

image_fn.build_config(image=docker_registry+ new_image,base_image='mlrun/ml-models-gpu',commands=['pip install fastai'])

image_fn.deploy(with_mlrun=True)
```

The image can now be used within the MLRun pipeline by specifying `image="docker-registry.default-tenant.app.groupwaretech.iguazio-c0.com:80/fastai"`

### MLRun Pipeline Development
The MLRun pipeline itself can be found within the notebook `MLRun.ipynb`. There is some boilerplate house-keeping code within the notebook itself to initialize an MLRun project and to actually launch the job on the cluster, but the relevant pipeline code can be found here.

Each of the pipeline components were imported into the project for use within the pipeline:
```python3
project.set_function('../components/yaml/get_data.yaml', 'get-data')
project.set_function('../components/yaml/create_data_bunches.yaml', 'create-data-bunches')
project.set_function('../components/yaml/hyper_lm.yaml', 'hyper-lm')
project.set_function('../components/yaml/train_lm.yaml', 'train-lm')
project.set_function('../components/yaml/hyper_clas.yaml', 'hyper-clas')
project.set_function('../components/yaml/train_clas.yaml', 'train-clas')
project.set_function('../components/yaml/model_server.yaml', 'model-server')
project.set_function('../components/yaml/model_server_tester.yaml', 'model-server-tester')
```

Finally, the pipeline itself uses each of these components to take the data as input, create a model using optimized hyper-parameters, and finally serve the model to an HTTP endpoint:
```python
from kfp import dsl
from mlrun import mount_v3io

funcs = {}

# Configure function resources and local settings
def init_functions(functions: dict, project=None, secrets=None):
    for f in functions.values():
        f.apply(mount_v3io())

# Create a Kubeflow Pipelines pipeline
@dsl.pipeline(
    name="Medical NLP Chemical Classification",
    description="Test Pipeline."
)
def kfpipeline(data_size=50000,
               split=0.5,
               hyper_lm_bs=[64, 128, 256],
               hyper_lm_drop_mult=[0.3, 0.6],
               hyper_lm_epochs=1,
               train_lm_epochs=10,
               hyper_clas_bs=[64, 128, 256],
               hyper_clas_thresh=[0.01],
               hyper_clas_drop_mult=[0.3, 0.6],
               hyper_clas_epochs=1,
               train_clas_epochs=10,
               model_endpoint_name="FASTAI_NLP",
               num_preds=8,
               num_tests=50):
    
    # Custom docker image with mlrun and fastai
    image = "docker-registry.default-tenant.app.groupwaretech.iguazio-c0.com:80/fastai"

    # Ingest the data set
    ingest = funcs['get-data'].as_step(
        name="get-data",
        handler='get_data',
        inputs={'data_size': data_size},
        outputs=['data'])
    
    # Create data bunches
    bunches = funcs['create-data-bunches'].as_step(
        name="create-data-bunches",
        handler='create_data_bunches',
        inputs={'data_path': ingest.outputs['data'], 'split' : split},
        outputs=['data_lm', 'data_clas'],
        image=image)
    
    # Language model Hyperparameters
    hyperparams = {"bs" : hyper_lm_bs,
                   "drop_mult" : hyper_lm_drop_mult}
    
    params = {"epochs" : hyper_lm_epochs,
              "num_samples" : data_size,
              "data_lm_path" : bunches.outputs['data_lm']}
    
    # Language model Hyperparameter tuning
    hyper_tune_lm = funcs['hyper-lm'].as_step(
        name="hyper-lm",
        handler='train_lm_model',
        params=params,
        hyperparams=hyperparams,
        selector='max.accuracy',
        outputs=['best_params'],
        image=image)
    
    # Language model training
    train_lm = funcs['train-lm'].as_step(
        name="train-lm",
        handler='train_lm',
        inputs={'train_lm_epochs': train_lm_epochs,
                'data_lm_path' : bunches.outputs['data_lm'],
                'num_samples' : data_size,
                'hyper_lm_best_params_path' : hyper_tune_lm.outputs['best_params']},
        outputs=['train_lm_model', 'train_lm_model_enc', 'train_lm_accuracy'],
        image=image)
    
    # Classification model Hyperparameters
    hyperparams = {"bs" : hyper_clas_bs,
                   "thresh" : hyper_clas_thresh,
                   "drop_mult" : hyper_clas_drop_mult}
    
    params = {"epochs" : hyper_clas_epochs,
              "num_samples" : data_size,
              "encodings" : train_lm.outputs['train_lm_model_enc'],
              "data_clas_path" : bunches.outputs['data_clas']}
    
    # Classification model Hyperparameter tuning
    hyper_tune_clas = funcs['hyper-clas'].as_step(
        name="hyper-clas",
        handler='train_clas_model',
        params=params,
        hyperparams=hyperparams,
        selector='max.fbeta',
        outputs=['best_params'],
        image=image)
    
    # Classification model training
    train_clas = funcs['train-clas'].as_step(
        name="train-clas",
        handler='train_clas',
        inputs={'train_clas_epochs': train_clas_epochs,
                'data_clas_path' : bunches.outputs['data_clas'],
                'num_samples' : data_size,
                'encodings' : train_lm.outputs['train_lm_model_enc'],
                'hyper_clas_best_params_path' : hyper_tune_clas.outputs['best_params']},
        outputs=['train_clas_model', 'train_clas_fbeta'],
        image=image)

    # Serve model
    deploy = funcs['model-server'].deploy_step(env={'DATA_CLAS_PATH' : bunches.outputs['data_clas'],
                                                   'MODEL_PATH' : train_clas.outputs['train_clas_model'],
                                                   f'SERVING_MODEL_{model_endpoint_name}': train_clas.outputs['train_clas_model'],
                                                   'NUM_PREDS' : num_preds})

    # Model serving tester
    tester = funcs['model-server-tester'].as_step(
        name='model-tester',
        inputs={'model_endpoint': deploy.outputs['endpoint'],
                'model_name' : model_endpoint_name,
                'data_size' : data_size,
                'data_path' : ingest.outputs['data'],
                'num_tests' : num_tests})
```

##### Note about Pipeline Component Workarounds
Within MLRun, one is able to store many things in the internal database including datasets, models, metrics, and artifacts (images, graphs, etc). This project does not use many of these capabilities due to some quirks with the FastAI library. The library requires the user to use the internal load functions to load models from a pickle. If you use the standard `pickle.load()` function instead, there is a strange recursion error. To work around this, pipeline components passed paths instead of actual objects.

Another workaround involved the model server component: I had to bypass the default `load` function and instead load during predictions (if it wasn't already loaded). The cause of this was that upon loading a model, the FastAI library prints out the name of the Wikipedia model it is using for transfer learning. MLRun interprets the console output as an error and kills the pod.

These are both quirks with the FastAI library and should not affect other libraries, but it is good to be aware of them.

### Viewing Pipeline Component Outputs
Logs for each pipeline component can be viewed using the `Logs` tab within the Kubeflow GUI or from the MLRun GUI. These are pulled directly from the container logs, meaning one could also view these logs using Kubernetes.

### Pipeline Runtime
A pipeline using 5000 data points takes about 22 minutes to run to completion.
A pipeline using 50000 data points takes about an hour and 14 minutes.

### Final Results
The final result of the pipeline leaves something to be desired in terms of accuracy, however it works in some respect. As a whole, this project was mainly meant to be a proof of concept and a roadmap for creating a Kubeflow Pipeline using MLRun and the Iguazio Data Platform.

Some of the predictions are more accurate than others, but there is certainly evidence that the model has learned relationships. For example, the following abstract about using fluorescent dyes to mark plasma cell membranes resulted in a prediction that included `Biomarkers, Tumor`, `Biomarkers`, and `Contrast Media`.
```
Text:  
 The lipophilic fluorescent probe diphenylhexatriene (DPH) has been shown previously to behave as a marker of plasma membrane in living cell systems, and it is therefore been widely used in membrane fluidity studies via fluorescence anisotropy measurements. The anisotropic coefficient, which is inversely related to the rotational motion of the probe in membrane phospholipids, was significantly higher at 37 degrees C than at 23 degrees C for 9 series of red blood cells ghosts obtained from three healthy subjects. We also have studied the importance of the nature of two different polaroid films which permits the observation of fluorescence polarization.  
Targets:  
 ['Fluorescent Dyes', 'Diphenylhexatriene']  
Preds:  
 b'["DNA", "Polymers", "Collagen", "Proteins", "Biomarkers, Tumor", "Antineoplastic Agents", "Biomarkers", "Contrast Media"]'
```

In this abstract, about the effects of Ciprofloxacinthe on soil microbial ecology and microbial ecosystem, the model correctly predicted `Anti-Bacterial Agents` and `Water Pollutants, Chemical`. Additionally, it predicted that both `Soil` and `Water` were relevant.
```
Text:
 While antibiotics are frequently found in the environment, their biodegradability and ecotoxicological effects are not well understood. Ciprofloxacin inhibits active and growing microorganisms and therefore can represent an important risk for the environment, especially for soil microbial ecology and microbial ecosystem services. We investigated the biodegradation of (14)C-ciprofloxacin in water and soil following OECD tests (301B, 307) to compare its fate in both systems. Ciprofloxacin is recalcitrant to biodegradation and transformation in the aqueous system. However, some mineralisation was observed in soil. The lower bioavailability of ciprofloxacin seems to reduce the compound's toxicity against microorganisms and allows its biodegradation. Moreover, ciprofloxacin strongly inhibits the microbial activities in both systems. Higher inhibition was observed in water than in soil and although its antimicrobial potency is reduced by sorption and aging in soil, ciprofloxacin remains biologically active over time. Therefore sorption does not completely eliminate the effects of this compound.
Targets:
 ['Anti-Bacterial Agents', 'Soil Pollutants', 'Water Pollutants, Chemical', 'Ciprofloxacin']
Preds:
 b'["Anti-Bacterial Agents", "Water Pollutants, Chemical", "Biomarkers", "Antineoplastic Agents", "Soil", "Antioxidants", "Oxygen", "Water"]'
```

The model that resulted in these predictions was trained with the following hyper-parameters:
```python
data_size=50000,
split=0.5,
hyper_lm_bs=[64, 128, 256],
hyper_lm_drop_mult=[0.3, 0.6],
hyper_lm_epochs=1,
train_lm_epochs=10,
hyper_clas_bs=[64, 128, 256],
hyper_clas_thresh=[0.01],
hyper_clas_drop_mult=[0.3, 0.6],
hyper_clas_epochs=1,
train_clas_epochs=10,
model_endpoint_name="FASTAI_NLP",
num_preds=8,
num_tests=50
```

A more exhaustive list of sample predictions has been included below:
```

```
<!--stackedit_data:
eyJoaXN0b3J5IjpbMTgwMDU1MzY4NywxODA4MjI5MjYyLC0xMz
AzNjAyODkzLDEyMDM2MDE1MTMsLTEzNTIyMjI2MTcsLTE1NTE1
MTQzMTEsNDAyODAwMjk5LC0xOTY1ODEwMjc5LC04NDE1Njg1MD
ksMTAzODIyMzkzNCwyNDg0MjI3MTEsMTA0NjY4NTM0NF19
-->