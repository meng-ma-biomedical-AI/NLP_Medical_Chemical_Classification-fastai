
# Relevant Medical Chemical Classification via NLP using FastAI

### Project Overview
This project aims to classify relevant chemicals from medical texts via Natural Language Processing using the FastAI library. The goal was to combine NLP, the medical domain, and automated machine learning pipelines using the Igauzio Data Platform.

The first step was to train a language model via transfer learning from a Wikipedia model to obtain language encodings that include the medical terminology. Using these language encodings. the next step was to train a classification model for the relevant chemicals per abstract.

The project is in the form on a Kubeflow Pipeline using MLRun and Nuclio on the Iguazio Data Platform. The pipeline includes:

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
The file `links.txt` was compiled using this sample. From there, `get_data.py` was used to download all the files via FTP using wget.
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
A pipeline using 5000 data points takes about 20 minutes to run to completion.
A pipeline using 50000 data points takes about an hour.

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
Text:  
 We have developed a unique male-sterility and fertility-restoration system in rice by combining Brassica napus cysteine-protease gene (BnCysP1) with anther-specific P12 promoter of rice for facilitating production of hybrid varieties. In diverse crop plants, male-sterility has been exploited as a useful approach for production of hybrid varieties to harness the benefits of hybrid vigour. The promoter region of Os12bglu38 gene of rice has been isolated from the developing panicles and was designated as P12. The promoter was fused with gusA reporter gene and was expressed in Arabidopsis and rice systems. Transgenic plants exhibited GUS activity in tapetal cells and pollen of the developing anthers indicating anther/pollen-specific expression of the promoter. For engineering nuclear male sterility, the coding region of Brassica napus cysteine protease1 (BnCysP1) was isolated from developing seeds and fused to P12 promoter. Transgenic rice plants obtained with P12-BnCysP1 failed to produce functional pollen grains. The F  
Targets:  
 ['Cysteine Proteases']  
Preds:  
 b'["Bacterial Proteins", "Recombinant Fusion Proteins", "RNA, Viral", "Recombinant Proteins", "Proteins", "DNA", "DNA-Binding Proteins", "Carrier Proteins"]'
Text:  
 Opioid misuse in the context of chronic opioid therapy (COT) is a growing concern. Depression may be a risk factor for opioid misuse, but it has been difficult to tease out the contribution of co-occurring substance abuse. This study aims to examine whether there is an association between depression and opioid misuse in patients receiving COT who have no history of substance abuse.  
Targets:  
 ['Analgesics, Opioid']  
Preds:  
 b'["Anti-Bacterial Agents", "Antineoplastic Agents", "Cytokines", "Biomarkers", "Antipsychotic Agents", "Anti-Inflammatory Agents, Non-Steroidal", "Anti-Inflammatory Agents", "Immunosuppressive Agents"]'
Text:  
 Compared to in situ vascular physiology where pro and anti-hemostatic processes are in balance to maintain hemostasis, the use of ECMO in a critically ill child increases the risk of hemorrhagic or thromboembolic events due to a perturbation in the balance inherent of this complex system. The ECMO circuit has pro-hemostatic effects due to contact activation of hemostasis and inflammatory pathways. In addition, the critical illness of the child can cause dysregulation of hemostasis that may shift between hyper and hypocoagulable states over time.  
Targets:  
 ['Anticoagulants', 'Hemostatics']  
Preds:  
 b'["Biomarkers", "Antineoplastic Agents", "Antipsychotic Agents", "Biomarkers, Tumor", "Anti-Inflammatory Agents, Non-Steroidal", "Insulin", "Anti-Inflammatory Agents", "Anti-Bacterial Agents"]'
Text:  
 Coincidence cloning allows the isolation of sequences held in common by two genomic DNA populations. Human DNA from two human-hamster hybrid cell lines was amplified by Alu-repeat primers (Alu PCR) and the products originating from the shared human chromosomal region were cloned. To achieve this, human sequences were amplified with very similar Alu primers from the two different human-hamster hybrid cell lines. The products were then digested with an appropriate restriction enzyme (either BamHI or Sal I), combined, denatured, and reannealed. The derived heteroduplex molecules (originating from the human regions common to both cell lines) had single BamHI and Sal I cohesive ends due to the primers used, so that they could be cloned in a double-digested plasmid vector. We used this method to enrich about 10-fold for Alu PCR products from the human chromosome 19q13.2 region, resulting in a region-specific clone collection. About 90% of the recombinants with BamHI-Sal I inserts are derived from the common region. This approach allows the boundaries for the regional probe isolation to be defined by combinations of hybrids rather than single hybrid cell lines, thus permitting greater flexibility in the selection of regions for probe isolation.  
Targets:  
 ['Oligonucleotide Probes', 'DNA']  
Preds:  
 b'["DNA", "RNA, Messenger", "DNA-Binding Proteins", "Bacterial Proteins", "DNA, Viral", "Genetic Markers", "Proteins", "DNA Primers"]'
Text:  
 The lipophilic fluorescent probe diphenylhexatriene (DPH) has been shown previously to behave as a marker of plasma membrane in living cell systems, and it is therefore been widely used in membrane fluidity studies via fluorescence anisotropy measurements. The anisotropic coefficient, which is inversely related to the rotational motion of the probe in membrane phospholipids, was significantly higher at 37 degrees C than at 23 degrees C for 9 series of red blood cells ghosts obtained from three healthy subjects. We also have studied the importance of the nature of two different polaroid films which permits the observation of fluorescence polarization.  
Targets:  
 ['Fluorescent Dyes', 'Diphenylhexatriene']  
Preds:  
 b'["DNA", "Polymers", "Collagen", "Proteins", "Biomarkers, Tumor", "Antineoplastic Agents", "Biomarkers", "Contrast Media"]'
Text:  
 An efficient method for the N-heterocyclic carbene (NHC)-catalyzed conjugate addition of acetyl anions to various α,β-unsaturated acceptors (Stetter reaction) has been optimized by using 2,3-butandione (biacetyl) as an alternative surrogate of acetaldehyde. The disclosed procedure proved to be compatible with microwave dielectric heating for reaction time reduction and with the use of different linear α-diketones as acyl anion donors (e.g. 3,4-hexanedione for propionyl anion additions). Moreover, the unprecedented umpolung reactivity of cyclic α-diketones in the atom economic nucleophilic acylation of chalcones is herein presented. Mechanistic aspects of the thiazolium-based catalysis involving linear and cyclic α-diketone substrates are also discussed.  
Targets:  
 ['Ketones', 'Thiazoles']  
Preds:  
 b'["Antineoplastic Agents", "Anti-Bacterial Agents", "Proteins", "Polymers", "Peptides", "Water", "Recombinant Proteins", "Amino Acids"]'
Text:  
 Detection, diagnosis and identification of Leishmaniasis may be difficult owing to low numbers of parasites present in clinical samples. The PCR has improved the sensitivity and specificity of diagnosis of several infectious diseases. A leishmania specific PCR assay was developed based on the SSUrRNA genes which amplifies DNA of all Leishmania species. Point mutations occurring within the rRNA genes allow differentiation of the Leishmania complexes using primers constructed with the 3/ ends complementary to the specific point mutations present in the SSU rRNA genes of the Leishmania species. Biopsy material, blood, lesion impressions and blood spots on filter paper can be used in the assay. In a longitudinal study on the incidence rates of VL, subclinical cases and PKDL in an endemic region of Sudan, filter paper blood spots from proven and suspected VL patients, PKDL and control samples from an endemic region in Sudan are being taken. The blood spots were analyzed in the DAT and by PCR and results compared with clinical and parasitological data. The first results indicate that the PCR on blood spots is a simple and sensitive means of detecting active VL; in PKDL patients parasites are detectable in the skin.  
Targets:  
 ['DNA, Kinetoplast', 'RNA, Protozoan', 'RNA, Ribosomal']  
Preds:  
 b'["Biomarkers, Tumor", "RNA, Messenger", "Antibodies, Monoclonal", "Antineoplastic Agents", "DNA, Viral", "DNA", "Biomarkers", "Genetic Markers"]'
Text:  
 Extended-spectrum β-lactamases (ESBLs) are enzymes capable of hydrolyzing oxyimino-β-lactams and inducing resistance to third generation cephalosporins. The genes encoding ESBLs are widespread and generally located on highly transmissible resistance plasmids. We aimed to investigate the complement of ESBL genes in E. coli and Klebsiella pneumoniae causing nosocomial infections in hospitals in Ho Chi Minh City, Vietnam.  
Targets:  
 ['Anti-Bacterial Agents', 'DNA, Bacterial', 'beta-Lactamases']  
Preds:  
 b'["Bacterial Proteins", "Recombinant Proteins", "Transcription Factors", "Anti-Bacterial Agents", "RNA, Messenger", "DNA Primers", "DNA-Binding Proteins", "DNA, Bacterial"]'
Text:  
 In the early 1990s, breakthrough discoveries on the genetics of Alzheimer's disease led to the identification of missense mutations in the amyloid-beta precursor protein gene. Research findings quickly followed, giving insights into molecular pathogenesis and possibilities for the development of new types of animal models. The complete toolbox of transgenic techniques, including pronuclear oocyte injection and homologous recombination, has been applied in the Alzheimer's disease field, to produce overexpressors, knockouts, knockins and regulatable transgenics. Transgenic models have dramatically advanced our understanding of pathogenic mechanisms and allowed therapeutic approaches to be tested. Following a brief introduction to Alzheimer's disease, various nontransgenic and transgenic animal models are described in terms of their values and limitations with respect to pathogenic, therapeutic and functional understandings of the human disease.  
Targets:  
 ['Amyloid beta-Peptides']  
Preds:  
 b'["RNA, Messenger", "Proteins", "MicroRNAs", "Transcription Factors", "DNA", "Antineoplastic Agents", "DNA-Binding Proteins", "Biomarkers"]'
Text:  
 Inbred mice with the mutation diabetes C57BL/KsJ db+/db+ and the mutation obese C57BL/6J ob/ob displayed a total liver mitochondrial capacity to oxidize glutamate or succinate which was approximately eight times greater than the capacity of the C57BL/6J +/+ control mice. This increase in oxidation capacity was estimated by multiplying the observed twofold increase in each of the following components: total liver weight, the mitochondrial protein content per gram of liver, and glutamate or succinate respiration activity per milligram of liver mitochondrial protein. No significant difference in liver mitochondrial function and capacity for oxidation was observed between db+/db+ and ob/ob mutants, which indicated that these results may be primarily mediated by the genetic factors responsible for obesity and hyperphagia in these mutants, and not by the genetic traits associated with diabetes. These findings may provide a biochemical foundation in support of the thrifty gene hypothesis.  
Targets:  
 ['Glutamates', 'Proteins', 'Succinates', 'Glutamic Acid', 'Succinic Acid']  
Preds:  
 b'["RNA, Messenger", "Recombinant Proteins", "Antineoplastic Agents", "Anti-Bacterial Agents", "Biomarkers", "Transcription Factors", "Insulin", "Blood Glucose"]'
Text:
 Human alveolar echinococcosis (AE) is caused by the fox tapeworm Echinococcus multilocularis and is usually lethal if left untreated. The current strategy for treating human AE is surgical resection of the parasite mass complemented by chemotherapy with benzimidazole compounds. However, reliable chemotherapeutic alternatives have not yet been developed stimulating the research of new treatment strategies such as the use of medicinal plants. The aim of the current study was to investigate the efficacy of the combination albendazole (ABZ)+thymol on mice infected with E. multilocularis metacestodes. For this purpose, mice infected with parasite material were treated daily for 20 days with ABZ (5 mg/kg), thymol (40 mg/kg) or ABZ (5 mg/kg)+thymol (40 mg/kg) or left untreated as controls. After mice were euthanized, cysts were removed from the peritoneal cavity and the treatment efficacy was evaluated by the mean cysts weight, viability of protoscoleces and ultrastructural changes of cysts and protoscoleces. The application of thymol or the combination of ABZ+thymol resulted in a significant reduction of the cysts weight compared to untreated mice. We also found that although ABZ and thymol had a scolicidal effect, the combination of the two compounds had a considerably stronger effect showing a reduction in the protoscoleces viability of 62%. These results were also corroborated by optical microscopy, SEM and TEM. Protoscoleces recovered from ABZ or thymol treated mice showed alterations as contraction of the soma region, rostellar disorganization and presence of blebs in the tegument. However both drugs when combined lead to a total loss of the typical morphology of protoscoleces. All cysts removed from control mice appeared intact and no change in ultrastructure was detected. In contrast, cysts developed in mice treated with ABZ revealed changes in the germinal layer as reduction in cell number, while the treatment with thymol or the ABZ+thymol combination predominantly showed presence of cell debris. On the other hand, no differences were found in alkaline phosphatase (AP), glutamate oxaloacetate transaminase (GOT) and glutamate pyruvate transaminase (GPT) activities between control and treated mice, indicating the lack of toxicity of the different drug treatments during the experiment. Because combined ABZ+thymol treatment exhibited higher treatment efficiency compared with the drugs applied separately against murine experimental alveolar echinococcosis, we propose it would be a useful option for the treatment of human AE.
Targets:
 ['Anthelmintics', 'Thymol', 'Albendazole']
Preds:
 b'["Anti-Bacterial Agents", "Antineoplastic Agents", "Biomarkers", "Antibodies, Monoclonal", "RNA, Messenger", "Recombinant Proteins", "Biomarkers, Tumor", "Immunoglobulin G"]'
Text:
 The anti-inflammatory and antibacterial mechanisms of bone marrow mesenchymal stem cells (MSCs) ameliorating lung injury in chronic obstructive pulmonary disease (COPD) mice induced by cigarette smoke and Haemophilus Parainfluenza (HPi) were studied. The experiment was divided into four groups in vivo: control group, COPD group, COPD+HPi group, and COPD+HPi+MSCs group. The indexes of emphysematous changes, inflammatory reaction and lung injury score, and antibacterial effects were evaluated in all groups. As compared with control group, emphysematous changes were significantly aggravated in COPD group, COPD+HPi group and COPD+HPi+MSCs group (P<0.01), the expression of necrosis factor-kappaB (NF-κB) signal pathway and proinflammatory cytokines in bronchoalveolar lavage fluid (BALF) were increased (P<0.01), and the phagocytic activity of alveolar macrophages was downregulated (P<0.01). As compared with COPD group, lung injury score, inflammatory cells and proinflammatory cytokines were significantly increased in the BALF of COPD+HPi group and COPD+HPi+MSCs group (P<0.01). As compared with COPD+HPi group, the expression of tumor necrosis factor-α stimulated protein/gene 6 (TSG-6) was increased, the NF-κB signal pathway was depressed, proinflammatory cytokine was significantly reduced, the anti-inflammatory cytokine IL-10 was increased, and lung injury score was significantly reduced in COPD+HPi+MSCs group. Meanwhile, the phagocytic activity of alveolar macrophages was significantly enhanced and bacterial counts in the lung were decreased. The results indicated cigarette smoke caused emphysematous changes in mice and the phagocytic activity of alveolar macrophages was decreased. The lung injury of acute exacerbation of COPD mice induced by cigarette smoke and HPi was alleviated through MSCs transplantation, which may be attributed to the fact that MSCs could promote macrophages into anti-inflammatory phenotype through secreting TSG-6, inhibit NF-кB signaling pathway, and reduce inflammatory response through reducing proinflammatory cytokines and promoting the expression of the anti-inflammatory cytokine. Simultaneously, MSCs could enhance phagocytic activity of macrophages and bacterial clearance. Meanwhile, we detected anti-inflammatory and antibacterial activity of macrophages regulated by MSCs in vitro. As compared with RAW264.7+HPi+CSE group, the expression of NF-кB p65, IL-1β, IL-6 and TNF-α was significantly reduced, and the phagocytic activity of macrophages was significantly increased in RAW264.7+HPi+CSE+MSCs group (P<0.01). The result indicated the macrophages co-cultured with MSCs may inhibit NF-кB signaling pathway and promote phagocytosis by paracrine mechanism.
Targets:
 ['Anti-Bacterial Agents', 'Anti-Inflammatory Agents']
Preds:
 b'["RNA, Messenger", "Cytokines", "Tumor Necrosis Factor-alpha", "Biomarkers", "NF-kappa B", "Reactive Oxygen Species", "Antioxidants", "Recombinant Proteins"]'
Text:
 Chronic Obstructive Pulmonary Disease (COPD) is a chronic inflammatory airway disease punctuated by exacerbations (AECOPD). Subjects with frequent AECOPD, defined by having at least two exacerbations per year, experience accelerated loss of lung function, deterioration in quality of life and increase in mortality. Fibroblast growth factor (FGF)23, a hormone associated with systemic inflammation and altered metabolism is elevated in COPD. However, associations between FGF23 and AECOPD are unknown. In this cross-sectional study, individuals with COPD were enrolled between June 2016 and December 2016. Plasma samples were analyzed for intact FGF23 levels. Logistic regression analyses were used to measure associations between clinical variables, FGF23, and the frequent exacerbator phenotype. Our results showed that FGF23 levels were higher in frequent exacerbators as compared to patients without frequent exacerbations. FGF23 was also independently associated with frequent exacerbations (OR 1.02; 95%CI 1.004-1.04; 
Targets:
 ['Fibroblast Growth Factors', 'fibroblast growth factor 23']
Preds:
 b'["Anti-Bacterial Agents", "Antineoplastic Agents", "Biomarkers", "Biomarkers, Tumor", "Immunosuppressive Agents", "Antiviral Agents", "Antibodies, Monoclonal", "C-Reactive Protein"]'
Text:
 Fatty acid synthase (FAS), a biosynthetic enzyme, normally functions in the liver to convert dietary carbohydrate to fat, but it is minimally expressed in most other normal adult tissues. FAS is expressed at markedly elevated levels in subsets of human breast, ovarian, and prostate carcinomas that are associated with poor prognoses. During the menstrual cycle, the expression of FAS in the human endometrium is closely linked to the expression of the proliferation antigen Ki-67, estrogen receptor (ER), and progesterone receptor (PR).
Targets:
 ['Ki-67 Antigen', 'Receptors, Estrogen', 'Receptors, Progesterone', 'Fatty Acid Synthases']
Preds:
 b'["RNA, Messenger", "Biomarkers", "Recombinant Proteins", "Transcription Factors", "Insulin", "Cytokines", "DNA-Binding Proteins", "Blood Glucose"]'
Text:
 Numerous biochemical experiments have invoked a model in which B-cell antigen receptor (BCR)-Fc receptor for immunoglobulin (Ig) G (FcgammaRII) coclustering provides a dominant negative signal that blocks B-cell activation. Here, we tested this model using quantitative confocal microscopic techniques applied to ex vivo splenic B cells. We found that FcgammaRII and BCR colocalized with intact anti-Ig and that the SH2 domain-containing inositol 5'-phosphatase (SHIP) was recruited to the same site. Colocalization of BCR and SHIP was inefficient in FcgammaRII-/- but not gamma chain-/- splenic B cells. We also examined the subcellular location of a variety of enzymes and adapter proteins involved in signal transduction. Several proteins (CD19, CD22, SHP-1, and Dok) and a lipid raft marker were co-recruited to the BCR, regardless of the presence or absence of FcgammaRII and SHIP. Other proteins (Btk, Vav, Rac, and F-actin) displayed reduced colocalization with BCR in the presence of FcgammaRII and SHIP. Colocalization of BCR and F-actin required phosphatidylinositol (PtdIns) 3-kinase and was inhibited by SHIP, because the block in BCR/F-actin colocalization was not seen in B cells of SHIP-/- animals. Furthermore, BCR internalization was inhibited with intact anti-Ig stimulation or by expression of a dominant-negative mutant form of Rac. From these results, we propose that SHIP recruitment to BCR/FcgammaRII and the resulting hydrolysis of PtdIns-3,4,5-trisphosphate prevents the appropriate spatial redistribution and activation of enzymes distal to PtdIns 3-kinase, including those that promote Rac activation, actin polymerization, and receptor internalization.
Targets:
 ['Actins', 'Receptors, IgG', 'ShPI proteinase inhibitor', 'Trypsin Inhibitor, Kunitz Soybean', 'Phosphatidylinositol 3-Kinases']
Preds:
 b'["RNA, Messenger", "DNA-Binding Proteins", "Transcription Factors", "Membrane Proteins", "Recombinant Proteins", "Recombinant Fusion Proteins", "Carrier Proteins", "Proteins"]'
Text:
 Rain gardens have been recommended as a best management practice to treat stormwater runoff. Replicate rain gardens were constructed in Haddam, CT, to treat roof runoff. The objective of this study was to assess whether the creation of a saturated zone in a rain garden improved retention of pollutants. The gardens were sized to store 2.54 cm (1 in) of runoff. Results show high retention of flow; only 0.8% overflowed. Overall, concentrations of nitrite+ nitrate-N, ammonia-N, and total-N (TN) in roof runoff were reduced significantly by the rain gardens. Total-P concentrations were significantly increased by both rain gardens. ANCOVA results show significant reductions in TN (18%) due to saturation. Redox potential also decreased in the saturated garden. Rain garden mulch was found to be a sink for metals, nitrogen, and phosphorus, but rain garden soils were a source for these pollutants. The design used for these rain gardens was effective for flow retention, but did not reduce concentrations of all pollutants even when modified. These findings suggest that high flow and pollutant retention could be achieved with the 2.54 cm design method, but the use of an underdrain could reduce overall pollutant retention.
Targets:
 ['Environmental Pollutants', 'Nitrates', 'Phosphorus', 'Lead', 'Copper', 'Zinc', 'Nitrogen']
Preds:
 b'["Water Pollutants, Chemical", "Anti-Bacterial Agents", "Soil", "Soil Pollutants", "Water", "Culture Media", "Oxygen", "Nitrogen"]'
Text:
 In some populations, complete shifts in the genotype of the strain of measles circulating in the population have been observed, with given genotypes being replaced by new genotypes. Studies have postulated that such shifts may be attributable to differences between the fitness of the new and the old genotypes.
Targets:
 ['Measles Vaccine']
Preds:
 b'["Biomarkers", "RNA, Messenger", "DNA", "Transcription Factors", "Insulin", "Anti-Bacterial Agents", "Genetic Markers", "DNA-Binding Proteins"]'
Text:
 Following experimental partial hepatectomy of 70% in the rat, there is a semisynchronized surge of hepatocyte proliferation that ceases after 48 to 72 hours. Little is known about the determinants governing the termination of the proliferative phase, although transforming growth factor (TGF) beta has been implicated as an important inhibitor of hepatocyte replication in this model. We previously reported an additional non-TGF-beta inhibitor in medium conditioned by nonparenchymal cells isolated from regenerating liver (CM-NPC-Reg) between 24 and 48 hours after partial hepatectomy, but it was not found in medium conditioned by nonparenchymal cells from unoperated control liver. CM-NPC-Reg suppressed replicative DNA synthesis of primary rat hepatocytes in response to hepatocyte growth factor (HGF), epidermal growth factor (EGF), or TGF-alpha as assessed by 3H-thymidine incorporation. We now present evidence that interleukin (IL)-1 is the major inhibitor of hepatocyte DNA synthesis present in CM-NPC-Reg. IL-1 receptor antagonist abrogated the inhibition, as did antibodies to rat IL-1alpha and -beta; a combination of both antibodies was required, implicating both IL-1alpha and IL-1beta as active constituents in CM-NPC-Reg. To investigate in vivo changes in IL-1 expression, we assessed expression of IL-1alpha messenger RNA (mRNA) in whole rat liver following partial hepatectomy; mRNA was down-regulated at 10 hours in the pre-replicative phase of liver regeneration and up-regulated at 24 hours and 48 hours when proliferation is waning. Rat hepatocytes isolated from liver 24 hours after partial hepatectomy showed increased sensitivity to the inhibitory action of IL-1. Exogenous IL-1beta, administered parenterally to a group of rats at 0 and 12 hours after partial hepatectomy significantly reduced the incorporation of the thymidine analogue, bromodeoxyuridine (BrdU), into hepatocytes at 18 hours. These data indicate that nonparenchymal cells isolated from regenerating rat liver elaborate IL-1, and support the hypothesis that IL-1 plays a role suppressing hepatocyte proliferation and terminating the surge of DNA synthesis induced after partial hepatectomy.
Targets:
 ['Culture Media, Conditioned', 'Insulin', 'Interleukin-1', 'RNA, Messenger', 'Receptors, Interleukin-1', 'Recombinant Proteins', 'Epidermal Growth Factor', 'DNA']
Preds:
 b'["RNA, Messenger", "Transcription Factors", "Cytokines", "DNA-Binding Proteins", "Tumor Necrosis Factor-alpha", "NF-kappa B", "Calcium", "Membrane Proteins"]'
Text:
 The glycosylation of alpha-dystroglycan (α-DG) is crucial in maintaining muscle cell membrane integrity. Dystroglycanopathies are identified by the loss of this glycosylation leading to a breakdown of muscle cell membrane integrity and eventual degeneration. However, a small portion of fibers expressing functionally glycosylated α-DG (F-α-DG) (revertant fibers, RF) have been identified. These fibers are generally small in size, centrally nucleated and linked to regenerating fibers. Examination of different muscles have shown various levels of RFs but it is unclear the extent of which they are present. Here we do a body-wide examination of muscles from the FKRP-P448L mutant mouse for the prevalence of RFs. We have identified great variation in the distribution of RF in different muscles and tissues. Triceps shows a large increase in RFs and together with centrally nucleated fibers whereas the pectoralis shows a reduction in revertant but increase in centrally nucleated fibers from 6 weeks to 6 months of age. We have also identified that the sciatic nerve with near normal levels of F-α-DG in the P448Lneo- mouse with reduced levels in the P448Lneo+ and absent in LARGEmyd. The salivary gland of LARGEmyd mice expresses high levels of F-α-DG. Interestingly the same glands in the P448Lneo-and to a lesser degree in P448Lneo+ also maintain considerable amount of F-α-DG, indicating the non-proliferating epithelial cells have a molecular setting permitting glycosylation.
Targets:
 ['Dag1 protein, mouse', 'Proteins', 'Dystroglycans', 'Fkrp protein, mouse', 'Transferases', 'Large1 protein, mouse', 'N-Acetylglucosaminyltransferases']
Preds:
 b'["RNA, Messenger", "DNA", "Recombinant Proteins", "Bacterial Proteins", "Peptides", "DNA Primers", "Membrane Proteins", "Proteins"]'
Text:
 The transition states and activation barriers of the 1,3-dipolar cycloadditions of azides with cycloalkynes and cycloalkenes were explored using B3LYP density functional theory (DFT) and spin component scaled SCS-MP2 methods. A survey of benzyl azide cycloadditions to substituted cyclooctynes (OMe, Cl, F, CN) showed that fluorine substitution has the most dramatic effect on reactivity. Azide cycloadditions to 3-substituted cyclooctynes prefer 1,5-addition regiochemistry in the gas phase, but CPCM solvation abolishes the regioselectivity preference, in accord with experiments in solution. The activation energies for phenyl azide addition to cycloalkynes decrease considerably as the ring size is decreased (cyclononyne DeltaG(double dagger) = 29.2 kcal/mol, cyclohexyne DeltaG(double dagger) = 14.1 kcal/mol). The origin of this trend is explained by the distortion/interaction model. Cycloalkynes are predicted to be significantly more reactive dipolarophiles than cycloalkenes. The activation barriers for the cycloadditions of phenyl azide and picryl azide (2,4,6-trinitrophenyl azide) to five- through nine-membered cycloalkenes were also studied and compared to experiment. Picryl azide has considerably lower activation barriers than phenyl azide. Dissection of the transition state energies into distortion and interaction energies revealed that "strain-promoted" cycloalkyne and cycloalkene cycloaddition transition states must still pay an energetic penalty to achieve their transition state geometries, and the differences in reactivity are more closely related to differences in distortion energies than the amount of strain released in the product. Trans-cycloalkene dipolarophiles have much lower barriers than cis-cycloalkenes.
Targets:
 ['Alkenes', 'Alkynes', 'Azides']
Preds:
 b'["Peptides", "Ligands", "Amino Acids", "Polymers", "Proteins", "Peptide Fragments", "DNA", "Water"]'
Text:
 Diffusion coefficients of sodium in barium borosilicate glasses having varying concentration of barium were determined by heterogeneous isotopic exchange method using (24)Na as the radiotracer for sodium. The measurements were carried out at various temperatures (748-798 K) to obtain the activation energy (E(a)) of diffusion. The E(a) values were found to increase with increasing barium content of the glass, indicating that introduction of barium in the borosilicate glass hinders the diffusion of alkali metal ions from the glass matrix. The results have been explained in terms of the electrostatic and structural factors, with the increasing barium concentration resulting in population of low energy sites by Na(+) ions and, plausibly, formation of more tight glass network. The leach rate measurements on the glass samples show similar trend.
Targets:
 ['Barium', 'Sodium']
Preds:
 b'["Polymers", "Water", "Biocompatible Materials", "Solutions", "Titanium", "Gels", "Membranes, Artificial", "Solvents"]'
Text:
 To study the possible implication of endogenous serotonin in the control of glucagon secretion in man, normal volunteers were subjected to alpha-cell stimulation before and after oral treatment with serotonin antagonists (cyproheptadine and methysergide) and with an inhibitor of serotonin synthesis (para-chlorophenylalanine, PCPA). After administration of cyproheptadine (16 mg daily, for two days) the glucagon responses to arginine (N=12) and to insulin-induced hypoglycemia (N=9) were more marked than in the control experiments (differences between maximal elevations: +165 pg/ml, P less than 0.0001, and +197 pg/ml, P less than 0.02, respectively). After methysergide treatment (9 mg daily, for two days), a potentiation of arginine-provoked glucagon secretion was also observed (+260 pg/ml, P less than 0.002; N=7). Similarly, after PCPA administration (2 g daily, for four days) the alpha-cell responsiveness to both aminogenic (N=12) and hypoglycemic (N=7) stimuli was enhanced (+108 pg/ml, P less than 0.05, and +164 pg/ml, P less than 0.05, respectively). Since glucagon secretion is potentiated by treatment with drugs which either antagonize serotonin action or inhibit its synthesis, the suggestion can be made that endogenous serotonin modulates alpha-cell function in man by acting as an inhibitor.
Targets:
 ['Insulin', 'Serotonin Antagonists', 'Cyproheptadine', 'Serotonin', 'Glucagon', 'Arginine', 'Fenclonine', 'Methysergide']
Preds:
 b'["Insulin", "Blood Glucose", "Norepinephrine", "Calcium", "Cyclic AMP", "Dopamine", "RNA, Messenger", "Acetylcholine"]'
Text:
 The multitubulin hypothesis holds that each tubulin isotype serves a unique role with respect to microtubule function. Here we investigate the role of the α-tubulin subunit Tuba1a in adult hippocampal neurogenesis and the formation of the dentate gyrus. Employing birth date labelling and immunohistological markers, we show that mice harbouring an S140G mutation in Tuba1a present with normal neurogenic potential, but that this neurogenesis is often ectopic. Morphological analysis of the dentate gyrus in adulthood revealed a disorganised subgranular zone and a dispersed granule cell layer. We have shown that these anatomical abnormalities are due to defective migration of prospero-homeobox-1-positive neurons and T-box-brain-2-positive progenitors during development. Such migratory defects may also be responsible for the cytoarchitectural defects observed in the dentate gyrus of patients with mutations in TUBA1A.
Targets:
 ['TUBA1A protein, human', 'Tubulin']
Preds:
 b'["RNA, Messenger", "Transcription Factors", "DNA-Binding Proteins", "Membrane Proteins", "Drosophila Proteins", "Nuclear Proteins", "Carrier Proteins", "Nerve Tissue Proteins"]'
Text:
 Hepatocellular carcinoma (HCC) is a leading cause of death worldwide. Among the surgical and nonsurgical treatments available, radiofrequency ablation (RFA) and sorafenib have been shown to have efficacy. There is little evidence whether combination of these therapies would have additional benefits.
Targets:
 ['Phenylurea Compounds', 'Niacinamide', 'Sorafenib']
Preds:
 b'["Biomarkers, Tumor", "Antineoplastic Agents", "Immunosuppressive Agents", "Biomarkers", "Adrenal Cortex Hormones", "Antibodies, Monoclonal, Humanized", "Angiogenesis Inhibitors", "Antibodies, Monoclonal"]'
Text:
 Postoperative analgesia using propacetamol was studied in 50 patients, 42 +/- 16 years old, after little or moderate surgery. Two grams of propacetamol in intravenous perfusion were administered every six hours. Three scales were utilized to note the intensity of the pain (simple verbal, behavioral and visual analogue scales), before the first injection and, one, four, six hours after. From this study, satisfactory analgesic efficiency and good tolerance of propacetamol were established.
Targets:
 ['Analgesics', 'Acetaminophen', 'propacetamol']
Preds:
 b'["Blood Glucose", "Insulin", "Oxygen", "Anti-Inflammatory Agents, Non-Steroidal", "Calcium", "Analgesics, Opioid", "Hydrocortisone", "Anticoagulants"]'
Text:
 The common channel theory suggests that bile reflux, through a common biliopancreatic channel, triggers acute pancreatitis. In the present study, this controversial issue was evaluated using an experimental model of hemorrhagic necrotizing pancreatitis.
Targets:
 ['Amylases']
Preds:
 b'["Biomarkers", "Insulin", "RNA, Messenger", "Transcription Factors", "DNA", "DNA-Binding Proteins", "Antineoplastic Agents", "Calcium"]'
Text:
 In this paper, we propose a new 3-D graphical representation of a DNA sequence and prove that it has two properties: (1) there is no circuit in the graph; (2) there exists a one-to-one correspondence between a DNA sequence and the graph. These properties guarantee it has nondegeneracy. Based on the 3-D graphical representation, we characterize a DNA sequence by a 12-dimensional vector whose components are normalized ALE-indexes of the corresponding L/L matrices. The proposed approach is tested by the phylogenetic analysis on three datasets, and the experimental assessment demonstrates its efficiency. 
Targets:
 ['DNA']
Preds:
 b'["DNA", "Proteins", "Bacterial Proteins", "Peptides", "Recombinant Proteins", "Ligands", "RNA, Messenger", "Membrane Proteins"]'
Text:
 Troponin I is a myofibrillar protein involved in the Ca(2+)-mediated regulation of actomyosin ATPase activity. We report here the isolation and characterization of the gene coding for the slow-muscle-specific isoform of the rat troponin I polypeptide (TpnI). Using restriction mapping, PCR mapping and partial DNA sequencing, we have determined the exon/intron arrangement of this gene. The transcription unit is 10.5-kb long and contains nine exons ranging in size from 4 bp to 330 bp. The rat TpnI(slow) gene is interrupted by large intervening sequences; a 3.3-kb intron separates the 5' untranslated exons from the protein-coding exons. Comparison of the structure of rat TpnI(slow) with that of quail TpnI(fast) reveals that they have a similar intron/exon organization. The 5' untranslated region of the rat gene contains an additional exon, otherwise, the positions of introns and coding exons map to essentially identical regions in both genes.
Targets:
 ['Troponin', 'Troponin I']
Preds:
 b'["RNA, Messenger", "DNA", "DNA-Binding Proteins", "Transcription Factors", "Recombinant Proteins", "Bacterial Proteins", "Proteins", "Carrier Proteins"]'
Text:
 Somatomedin-C or insulin-like growth factor I (Sm-C/IGF-I) and insulin-like growth factor II (IGF-II) have been implicated in the regulation of fetal growth and development. In the present study 32P-labeled complementary DNA probes encoding human and mouse Sm-C/IGF-I and human IGF-II were used in Northern blot hybridizations to analyse rat Sm-C/IGF-I and IGF-II mRNAs in poly(A+) RNAs from intestine, liver, lung, and brain of adult rats and fetal rats between day 14 and 17 of gestation. In fetal rats, all four tissues contained a major mRNA of 1.7 kilobases (kb) that hybridized with the human Sm-C/IGF-I cDNA and mRNAs of 7.5, 4.7, 1.7, and 1.2 kb that hybridized with the mouse Sm-C/IGF-I cDNA. Adult rat intestine, liver, and lung also contained these mRNAs but Sm-C/IGF-I mRNAs were not detected in adult rat brain. These findings provide direct support for prior observations that multiple tissues in the fetus synthesize immunoreactive Sm-C/IGF-I and imply a role for Sm-C/IGF-I in fetal development as well as postnatally. The abundance of a 7.5-kb Sm-C/IGF-I mRNA in poly(A+) RNAs from adult rat liver was 10-50-fold higher than in other adult rat tissues which provides further evidence that in the adult rat the liver is a major site of Sm-C/IGF-I synthesis and source of circulating Sm-C/IGF-I. Multiple IGF-II mRNAs of estimated sizes 4.7, 3.9, 2.2, 1.75, and 1.2 kb were observed in fetal rat intestine, liver, lung, and brain. The 4.7- and 3.9-kb mRNAs were the major hybridizing IGF-II mRNAs in all fetal tissues. Higher abundance of IGF-II mRNAs in rat fetal tissues compared with adult tissues supports prior hypotheses, based on serum IGF-II concentrations, that IGF-II is predominantly a fetal somatomedin. IGF-II mRNAs are present, however, in some poly(A+) RNAs from adult rat tissues. The brain was the only tissue in the adult rat where the 4.7- and 3.9-kb IGF-II mRNAs were consistently detected. Some samples of adult rat intestine contained the 4.7- and 3.9-kb IGF-II mRNAs and some samples of adult liver and lung contained the 4.7-kb mRNA. These findings suggest that a role for IGF-II in the adult rat, particularly in the central nervous system, cannot be excluded.
Targets:
 ['RNA, Messenger', 'Somatomedins', 'Insulin-Like Growth Factor I', 'Insulin-Like Growth Factor II', 'DNA']
Preds:
 b'["RNA, Messenger", "Transcription Factors", "Recombinant Proteins", "DNA, Complementary", "DNA-Binding Proteins", "DNA", "DNA Primers", "Membrane Proteins"]'
Text:
 A new fluorescent sensor consisting of Cd(II)-cylcen appended aminocoumarin and a substrate peptide for protein kinase A (PKA) has been designed. Upon phosphorylation by PKA, the metal complex moiety binds to a phosphorylated residue, which in turn displaces the coumarin fluorophore, and this event results in ratiometric change of excitation spectrum in neutral aqueous solution.
Targets:
 ['Anions', 'Coumarins', 'Fluorescent Dyes', 'Peptides', 'Cadmium', 'Water', 'Cyclic AMP-Dependent Protein Kinases']
Preds:
 b'["Ligands", "DNA", "Proteins", "Macromolecular Substances", "Adenosine Triphosphate", "Peptides", "Recombinant Proteins", "Membrane Proteins"]'
Text:
 The nucleolar Saccharomyces cerevisiae protein Nep1 was previously shown to bind to a specific site of the 18S rRNA and to be involved in assembly of Rps19p into pre-40S ribosome subunits. Here we report on the identification of tma23 and nop6 mutations as recessive suppressors of a nep1(ts) mutant allele and the nep1 deletion as well. Green fluorescent protein fusions localized Tma23p and Nop6p within the nucleolus, indicating their function in ribosome biogenesis. The high lysine content of both proteins and an RNA binding motif in the Nop6p amino acid sequence suggest RNA-binding functions for both factors. Surprisingly, in contrast to Nep1p, Tma23p and Nop6p seem to be specific for fungi as no homologues could be found in higher eukaryotes. In contrast to most other ribosome biogenesis factors, Tma23p and Nop6p are nonessential in S. cerevisiae. Interestingly, the tma23 mutants showed a considerably increased resistance against the aminoglycoside G418, probably due to a structural change in the 40S ribosomal subunit, which could be the result of incorrectly folded 18S rRNA gene, missing rRNA modifications or the lack of a ribosomal protein.
Targets:
 ['EMG1 protein, S cerevisiae', 'Nop6 protein, S cerevisiae', 'Nuclear Proteins', 'RNA-Binding Proteins', 'Ribosomal Proteins', 'Saccharomyces cerevisiae Proteins', 'Tma23 protein, S cerevisiae']
Preds:
 b'["RNA, Messenger", "Transcription Factors", "DNA-Binding Proteins", "DNA Primers", "Bacterial Proteins", "DNA", "Fungal Proteins", "Recombinant Proteins"]'
Text:
 Exhaled breath condensate (EBC) contains among a large number of mediators hydrogen peroxide (H2O2) as a marker of airway inflammation and oxidative stress. Similarly EBC pH also changes in respiratory diseases. It was the aim of our investigation to prove if hydrogen peroxide release and changes in pH of EBC changes with exercise.
Targets:
 ['Hydrogen Peroxide']
Preds:
 b'["Biomarkers", "Antioxidants", "Insulin", "RNA, Messenger", "Calcium", "Glucose", "Blood Glucose", "Oxygen"]'
Text:
 Abstract Tirasemtiv is a fast skeletal muscle activator that increases the sensitivity of the sarcomere to calcium, increasing the efficiency of muscle contraction when the muscle is stimulated at submaximal contraction frequencies. A previous study showed single doses of tirasemtiv to be well tolerated and associated with potentially important improvements in a variety of functional outcomes. This study determined safety of tirasemtiv when given at doses up to 500 mg daily for three weeks. Tirasemtiv was given as a single daily dose up to 375 mg for two weeks, with and without concomitant riluzole. In a separate cohort, an ascending dose protocol evaluated a total dose of 500 mg daily given in two divided doses. Safety and tolerability were assessed, as well as measures of function, muscle strength and endurance. Results showed that tirasemtiv was well tolerated, with dizziness the most common adverse event. Tirasemtiv approximately doubled the serum concentration of riluzole. Trends were noted for improvement in ALSFRS-R, Maximum Minute Ventilation, and Nasal Inspiratory Pressure. In conclusion, tirasemtiv is well tolerated and can be given safely with a reduced dose of riluzole. Positive trends in multiple exploratory outcome measures support the further study of this agent in ALS. 
Targets:
 ['CK-2017357', 'Imidazoles', 'Pyrazines']
Preds:
 b'["Insulin", "Blood Glucose", "Calcium", "Oxygen", "Biomarkers", "Norepinephrine", "Sodium", "Anti-Inflammatory Agents, Non-Steroidal"]'
Text:
 An exaggerated release of inflammatory mediators has been implicated in the pathogenesis of necrotizing enterocolitis (NEC). Oral administration of a human immunoglobulin preparation (serum IgA-IgG) has been demonstrated to be an effective prophylaxis for NEC. The aim of the present study was to examine the regulatory effect of a human IgA-IgG preparation on the release of inflammatory cytokines in human monocytes. Our results indicate that the immunoglobulin preparation inhibits TNF-alpha and IL-6 release in monocytes following stimulation with heat-inactivated Hib in a dose-dependent manner. This might have a biological relevance in infants receiving oral immunoglobulin prophylaxis for NEC, since modulation of the release of inflammatory mediators at the level of the gastrointestinal mucosa could interfere with the development of noxious sequelae of acute and/or chronic inflammation initiated by microbial pathogens or their toxins that finally lead to the pathologic changes associated with NEC.
Targets:
 ['Cytokines', 'Immunoglobulin A', 'Immunoglobulin G', 'Immunoglobulins', 'Interleukin-6', 'Tumor Necrosis Factor-alpha']
Preds:
 b'["Cytokines", "Biomarkers", "Anti-Bacterial Agents", "RNA, Messenger", "Tumor Necrosis Factor-alpha", "Antineoplastic Agents", "Antibodies, Monoclonal", "Recombinant Proteins"]'
Text:
 Innovative approaches are needed to support patients' adherence to drug therapy. The Real Time Medication Monitoring (RTMM) system offers real time monitoring of patients' medication use combined with short message service (SMS) reminders if patients forget to take their medication. This combination of monitoring and tailored reminders provides opportunities to improve adherence. This article describes the design of an intervention study aimed at evaluating the effect of RTMM on adherence to oral antidiabetics.
Targets:
 ['Hypoglycemic Agents']
Preds:
 b'["Antineoplastic Agents", "Pharmaceutical Preparations", "Antidepressive Agents", "Biomarkers", "Analgesics, Opioid", "Analgesics", "Contrast Media", "Psychotropic Drugs"]'
Text:
 Electrical stimulation (ES) is therapeutic to many bone diseases, from promoting fracture regeneration to orthopedic intervention. The application of ES offers substantial therapeutic potential, while optimal ES parameters and the underlying mechanisms responsible for the positive clinical impact are poorly understood. In this study, we assembled an ES cell culture and monitoring device. Mc-3T3-E1 cells were subjected to different frequency to investigate the effect of osteogenesis. Cell proliferation, DNA synthesis, the mRNA levels of osteosis-related genes, the activity of alkaline phosphatase (ALP), and intracellular concentration of Ca2+ were thoroughly evaluated. We found that 100 Hz could up-regulate the mRNA levels of collagen I, collagen II and Runx2. On the contrary, ES could down-regulate the mRNA levels of osteopontin (OPN). ALP activity assay and Fast Blue RR salt stain showed that 100 Hz could accelerate cells differentiation. Compared to the control group, 100 Hz could promote cell proliferation. Furthermore, 1 Hz to 10 Hz could improve calcium deposition in the intracellular matrix. Overall, these results indicate that 100Hz ES exhibits superior potentialities in osteogenesis, which should be beneficial for the clinical applications of ES for the treatment of bone diseases.
Targets:
 ['Collagen Type I', 'Osteopontin', 'Alkaline Phosphatase', 'Calcium']
Preds:
 b'["Collagen", "Antineoplastic Agents", "Calcium", "RNA, Messenger", "Biomarkers", "Recombinant Proteins", "Cytokines", "Oxygen"]'
Text:
 While antibiotics are frequently found in the environment, their biodegradability and ecotoxicological effects are not well understood. Ciprofloxacin inhibits active and growing microorganisms and therefore can represent an important risk for the environment, especially for soil microbial ecology and microbial ecosystem services. We investigated the biodegradation of (14)C-ciprofloxacin in water and soil following OECD tests (301B, 307) to compare its fate in both systems. Ciprofloxacin is recalcitrant to biodegradation and transformation in the aqueous system. However, some mineralisation was observed in soil. The lower bioavailability of ciprofloxacin seems to reduce the compound's toxicity against microorganisms and allows its biodegradation. Moreover, ciprofloxacin strongly inhibits the microbial activities in both systems. Higher inhibition was observed in water than in soil and although its antimicrobial potency is reduced by sorption and aging in soil, ciprofloxacin remains biologically active over time. Therefore sorption does not completely eliminate the effects of this compound.
Targets:
 ['Anti-Bacterial Agents', 'Soil Pollutants', 'Water Pollutants, Chemical', 'Ciprofloxacin']
Preds:
 b'["Anti-Bacterial Agents", "Water Pollutants, Chemical", "Biomarkers", "Antineoplastic Agents", "Soil", "Antioxidants", "Oxygen", "Water"]'
Text:
 Incubation of gonococci under conditions optimal for autolysis resulted in increased sensitivity and enhancement of the coagglutination reaction of the Phadebact gonococcus test. These conditions included an alkaline pH (pH 8.3) and the presence of divalent cation chelators such as ethylenediaminetetraacetic acid or ethylene glycol-bis(beta-aminoethyl ether)-N,N-tetraacetic acid. Heating cell suspensions at 90 degrees C for 15 min before assay by coagglutination produced a further increase in sensitivity and enhancement of the reaction. Gonococcal lipopolysaccharide was found to be an important antigen in these coagglutination reactions. The detection of lipopolysaccharide was markedly enhanced by the addition of chelating agents.
Targets:
 ['Antigens, Bacterial', 'Ethylene Glycols', 'Lipopolysaccharides', 'Egtazic Acid', 'Edetic Acid']
Preds:
 b'["Calcium", "Oxygen", "Water", "Glucose", "Sodium", "Antioxidants", "Carbon Dioxide", "Potassium"]'
Text:
 The purpose of this study was to compare intraosseous graft healing between the doubled flexor tendon (FT) graft and the bone-patellar tendon-bone (BPTB) graft in anterior cruciate ligament (ACL) reconstruction.
Targets:
 ['Collagen']
Preds:
 b'["Biomarkers", "Recombinant Proteins", "Anti-Bacterial Agents", "Collagen", "Calcium", "Oxygen", "RNA, Messenger", "Blood Glucose"]'
Text:
 Selection of human cells for resistance to vincristine or doxorubicin often induces overexpression of the multidrug resistance 1 gene (MDR1), which encodes the cell surface P-glycoprotein, as a result of gene amplification or transcriptional activation. Moreover, overexpression of the MDR1 gene has been shown to be associated closely with clinical outcome in various hematological malignancies, including acute myeloid leukemia (AML). However, the precise mechanism underlying overexpression of the MDR1 gene during acquisition of drug resistance remains unclear. We recently described an inverse correlation between the methylation status of CpG sites at the promoter region and expression of the MDR1 gene in malignant cell lines. In this study, we expanded this analysis to 42 clinical AML samples. We adapted a quantitative reverse transcription-polymerase chain reaction (RT-PCR) assay for gene expression and a quantitative PCR after digestion by Hpa II for methylation status of the MDR1 gene. We observed a statistically significant inverse correlation between methylation and MDR1 expression in clinical samples. The hypomethylation status of the MDR1 promoter region might be a necessary condition for MDR1 gene overexpression and establishment of P-glycoprotein-mediated multidrug resistance in AML patients.
Targets:
 ['ATP Binding Cassette Transporter, Subfamily B, Member 1']
Preds:
 b'["RNA, Messenger", "DNA-Binding Proteins", "Transcription Factors", "Tumor Suppressor Protein p53", "RNA, Small Interfering", "NF-kappa B", "Antineoplastic Agents", "Biomarkers, Tumor"]'
Text:
 The aim of this study was to assess the relationship between the two types of posttranslational modifications of proteins in RA: glycosylation on the example of carbohydrate-deficient transferrin and citrullination by means of autoantibodies to cyclic citrullinated peptides.
Targets:
 ['Anti-Citrullinated Protein Antibodies', 'Biomarkers', 'Peptides, Cyclic', 'Transferrin', 'carbohydrate-deficient transferrin', 'cyclic citrullinated peptide', 'Rheumatoid Factor']
Preds:
 b'["RNA, Messenger", "DNA", "Transcription Factors", "Bacterial Proteins", "DNA-Binding Proteins", "Proteins", "Recombinant Proteins", "DNA Primers"]'
Text:
 The purpose of the present study was to examine the effectiveness of fluorine and silver ions implanted and deposited into acrylic resin (poly(methyl methacrylate)) using a hybrid process of plasma-based ion implantation and deposition. The surface characteristics were evaluated by X-ray photoelectron spectroscopy (XPS), contact angle measurements, and atomic force microscopy. In addition, an antibacterial activity test was performed by the adenosine-5'-triphosphate luminescence method. XPS spectra of modified specimens revealed peaks due to fluoride and silver. The water contact angle increased significantly due to implantation and deposition of both fluorine and silver ions. In addition, the presence of fluorine and silver was found to inhibit bacterial growth. These results suggest that fluorine and silver dual-ion implantation and deposition can provide antibacterial properties to acrylic medical and dental devices.
Targets:
 ['Anti-Bacterial Agents', 'Ions', 'Resins, Synthetic', 'Fluorine', 'Silver', 'Polymethyl Methacrylate']
Preds:
 b'["Polymers", "Biocompatible Materials", "Titanium", "Polyethylene Glycols", "Hydrogels", "Drug Carriers", "Water", "Collagen"]'
Text:
 ABSTRACT The goal of this study is to determine whether dermal fibroblasts lacking syndecan-1 (sdc1) show differences in integrin expression and function that could contribute to the delayed skin and corneal wound healing phenotypes seen in sdc-1 null mice. Using primary dermal fibroblasts, we show that after 3 days in culture no differences in alpha-smooth muscle actin were detected but sdc-1 null cells expressed significantly more alphav and beta1 integrin than wildtype (wt) cells. Transforming growth factor beta1 (TGFbeta1) treatment at day 3 increased alphav- and beta1-integrin expression in sdc-1 null cells at day 5 whereas wt cells showed increased expression only of alphav-integrin. Using time-lapse studies, we showed that the sdc-1 null fibroblasts migrate faster than wt fibroblasts, treatment with TGFbeta1 increased these migration differences, and treatment with a TGFbeta1 antagonist caused sdc-1 null fibroblasts to slow down and migrate at the same rate as untreated wt cells. Cell spreading studies on replated fibroblasts showed altered cell spreading and focal adhesion formation on vitronectin and fibronectin-coated surfaces. Additional time lapse studies with beta1- and alphav-integrin antibody antagonists, showed that wt fibroblasts expressing sdc-1 had activated integrins on their surface that impeded their migration whereas the null cells expressed alphav-containing integrins which were less adhesive and enhanced cell migration. Surface expression studies showed increased surface expression of alpha2beta1 and alpha3beta1 on the sdc-1 null fibroblasts compared with wt fibroblasts but no significant differences in surface expression of alpha5beta1, alphavbeta3, or alphavbeta5. Taken together, our data indicates that sdc-1 functions in the activation of alphav-containing integrins and support the hypothesis that impaired wound healing phenotypes seen in sdc-1 null mice could be due to integrin-mediated defects in fibroblast migration after injury.
Targets:
 ['Actins', 'Fibronectins', 'Integrin alphaV', 'Integrin beta1', 'Sdc1 protein, mouse', 'Syndecan-1', 'Transforming Growth Factor beta1']
Preds:
 b'["RNA, Messenger", "Transcription Factors", "DNA-Binding Proteins", "Membrane Proteins", "DNA", "Nuclear Proteins", "Recombinant Proteins", "DNA Primers"]'
Text:
 The aim of the current study was to compare sperm quality characteristics of the collared peccary (Pecari tajacu) following freezing in extenders supplemented with whole egg yolk and different concentrations of low-density lipoproteins (LDL). Semen from 11 adult males was obtained by electroejaculation and evaluated for sperm motility, vigor, morphology as well as membrane integrity analyzed by the hypo-osmotic swelling (HOS) test and a fluorescent staining. Moreover, the semen was diluted in a Tris-based extender containing 20% egg yolk (control group) or 5, 10 or 20% LDL (treatment groups). The semen samples were frozen in liquid nitrogen and thawed in a water bath for 60s at 37°C. The treatments did not affect (p>0.05) sperm vigor, morphology or membrane integrity analyzed by the HOS test. However, post-thaw sperm motility was significantly higher (p<0.05) in the extender supplemented with 20% LDL (36.4 ± 5.3%) compared with the egg yolk extender and extender supplemented with 10% LDL. Furthermore, the percentage of membrane-intact frozen-thawed spermatozoa analyzed by the fluorescent staining was significantly higher (p<0.05) in the extender supplemented with 20% LDL (27.4 ± 6.5%) than in the other groups. In conclusion, 20% LDL can be used to substitute the whole egg yolk as a cryoprotective additive for freezing semen of the collared peccary.
Targets:
 ['Cryoprotective Agents', 'Lipoproteins, LDL']
Preds:
 b'["Water Pollutants, Chemical", "Water", "Blood Glucose", "Oxygen", "Glucose", "Lipids", "Amino Acids", "Nitrogen"]'
Text:
 The application is in the field of adult neurogenesis and its therapeutic potential. It aims to characterize the activity of apigenin and related compounds on adult neurogenesis in vivo and in vitro. Apigenin and related compounds are derivatives used in food products. They were administered intraperitoneally and orally in adult rodents and assessed for their activity in promoting the generation of neuronal cells and learning and memory performance. They were also tested on adult rat hippocampal-derived neural progenitor and stem cells to assess their neurogenic property. Apigenin and related compounds stimulate adult neurogenesis in vivo and in vitro, by promoting neuronal differentiation. Apigenin promotes learning and memory performance in the Morris water task. The application claims the use of apigenin and related compounds for stimulating adult neurogenesis and for the treatment of neurological diseases, disorders and injuries, by stimulating the generation of neuronal cells in the adult brain.
Targets:
 ['Apigenin']
Preds:
 b'["Cytokines", "Biomarkers", "Transcription Factors", "Dopamine", "RNA, Messenger", "Antineoplastic Agents", "Insulin", "MicroRNAs"]'
Text:
 In a biochemical reaction there is generally a change in the binding of hydrogen ions and metal ions. Therefore, calorimetric measurements of enthalpies of reaction have to be adjusted for the enthalpies of reaction of the hydrogen ions and metal ions produced or consumed with the buffer. It can be shown that this yields the standard transformed enthalpy of reaction that determines the change in the apparent equilibrium constant K' (written in terms of sums of concentrations of species of a reactant) with temperature at the chosen pH and concentration of free metal ion. The derivations are based on the assumption that the changes in pH and free metal ion concentrations in the calorimetric experiment are small. This assumption is experimentally realized if a solution is well buffered for hydrogen and metal ions. The derived equations are discussed in terms of the implications they have for the performance and interpretation of calorimetric measurements.
Targets:
 ['Magnesium']
Preds:
 b'["Water", "Polymers", "Proteins", "DNA", "Ligands", "Biocompatible Materials", "Amino Acids", "Titanium"]'
Text:
 The present work studied the effect of preoptic catecholamine on acupuncture analgesia. The catecholaminergic terminals were destructed by microinjection of 6-hydroxydopamine into the preoptic area and the destruction was checked by fluorescence histochemical method. The results showed that the destruction of catecholaminergic terminals significantly enhance acupuncture analgesia, suggesting that the reduction of catecholamine content in the preoptic area may enhance acupuncture analgesia.
Targets:
 ['Hydroxydopamines', 'Receptors, Adrenergic', 'Receptors, Dopamine', 'Oxidopamine', 'Dopamine', 'Norepinephrine']
Preds:
 b'["Insulin", "RNA, Messenger", "Calcium", "Dopamine", "Blood Glucose", "Biomarkers", "Glucose", "Norepinephrine"]'
Text:
 Patients with a high-output stoma (HOS) (> 2000 ml/day) suffer from dehydration, hypomagnesaemia and under-nutrition. This study aimed to determine the incidence, aetiology and outcome of HOS.
Targets:
 ['Magnesium']
Preds:
 b'["Biomarkers", "Blood Glucose", "Insulin", "Anti-Bacterial Agents", "Oxygen", "Anticoagulants", "Antihypertensive Agents", "Adrenal Cortex Hormones"]'
Text:
 Left-sided displacement of the abomasum (LDA) is a common disease in many dairy cattle breeds. A genome-wide screen for QTL for LDA in German Holstein (GH) cows indicated motilin (MLN) as a candidate gene on bovine chromosome 23. Genomic DNA sequence analysis of MLN revealed a total of 32 polymorphisms. All informative polymorphisms used for association analyses in a random sample of 1,136 GH cows confirmed MLN as a candidate for LDA. A single nucleotide polymorphism (FN298674:g.90T>C) located within the first non-coding exon of bovine MLN affects a NKX2-5 transcription factor binding site and showed significant associations (OR(allele) = 0.64; -log(10)P(allele) = 6.8, -log(10)P(genotype) = 7.0) with LDA. An expression study gave evidence of a significantly decreased MLN expression in cows carrying the mutant allele (C). In individuals heterozygous or homozygous for the mutation, MLN expression was decreased by 89% relative to the wildtype. FN298674:g.90T>C may therefore play a role in bovine LDA via the motility of the abomasum. This MLN SNP appears useful to reduce the incidence of LDA in German Holstein cattle and provides a first step towards a deeper understanding of the genetics of LDA.
Targets:
 ['Transcription Factors', 'Motilin']
Preds:
 b'["RNA, Messenger", "DNA", "DNA-Binding Proteins", "Recombinant Proteins", "Transcription Factors", "DNA Primers", "Biomarkers", "Membrane Proteins"]'
Text:
 For the last 40 yr, the first line of treatment for anovulation in infertile women has been clomiphene citrate (CC). CC is a safe, effective oral agent but is known to have relatively common antiestrogenic endometrial and cervical mucous side effects that could prevent pregnancy in the face of successful ovulation. In addition, there is a significant risk of multiple pregnancy with CC, compared with natural cycles. Because of these problems, we proposed the concept of aromatase inhibition as a new method of ovulation induction that could avoid many of the adverse effects of CC. The objective of this review was to describe the different physiological mechanisms of action for CC and aromatase inhibitors (AIs) and compare studies of efficacy for both agents for ovulation induction.
Targets:
 ['Aromatase Inhibitors', 'Clomiphene']
Preds:
 b'["Biomarkers", "Anti-Bacterial Agents", "Antineoplastic Agents", "Cytokines", "Antibodies, Monoclonal", "Immunoglobulin G", "Biomarkers, Tumor", "Estrogens"]'
Text:
 The Akt/CREB signalling pathway is involved in neuronal survival and protection. Autophagy is also likely to be involved in survival mechanisms. Nimodipine is an L-type calcium channel antagonist that reduces excessive calcium influx during pathological conditions (contributing to its neuroprotective properties). However, the potential role of nimodipine in autophagic and Akt/CREB signalling is not well understood. In addition, little is known about the relationship between autophagic and Akt/CREB signalling. Here, we designed a way to evaluate these issues. Adult male Sprague-Dawley rats were subjected to permanent bilateral occlusion of the common carotid artery (2VO) and randomly divided into three groups: the Vehicle (2VO), Nimodipine10 (2VO+nimodipine 10mg/kg), and Nimodipine20 (2VO+nimodipine 20mg/kg) groups. A fourth group of animals served as Sham controls. Each group was investigated at 4 and 8 weeks post-operatively and assessed using the Morris water maze. Nimodipine significantly alleviated spatial learning and memory impairments and inhibited the loss of neurons in the CA1 region of the hippocampus. These drug effects were more pronounced at 8 weeks than at 4 weeks. The activities of LC3 II p-Akt and p-CREB were examined using immunohistochemistry and western blotting. Suppressing autophagy induced pyramidal cell death without affecting increased pro-survival signalling induced by nimodipine. Nimodipine protected the brain from chronic cerebral hypoperfusion by activating the Akt/CREB signalling pathway. Autophagy has a neuroprotective effect on rats after 2VO. Autophagy is likely part of an integrated survival signalling network involving the Akt/CREB pathway.
Targets:
 ['CREB1 protein, rat', 'Calcium Channel Blockers', 'Cyclic AMP Response Element-Binding Protein', 'LC3 protein, rat', 'Microtubule-Associated Proteins', 'Neuroprotective Agents', 'Nimodipine', 'Proto-Oncogene Proteins c-akt']
Preds:
 b'["RNA, Messenger", "Proto-Oncogene Proteins c-akt", "Transcription Factors", "NF-kappa B", "RNA, Small Interfering", "Nerve Tissue Proteins", "Calcium", "Membrane Proteins"]'
Text:
 Several bone resorptive stimuli affect osteoclasts indirectly by modulating the production and release of osteoblastic factors. Using electrophoretic mobility shift assays, we found that not only tumour necrosis factor-alpha (TNF-alpha) but also interleukin-1beta and parathyroid hormone (PTH) caused dose and time-related increases in nuclear factor kappaB (NF-kappaB)-DNA binding in Saos-2 human osteoblastic (hOB) cells. Activation of NF-kappaB by TNF-alpha was reproduced in primary hOBs. In contrast, consistent with their previously reported lack of response to steroid hormones, Saos-2 cells did not respond to 1,25-dihydroxyvitamin D(3). We suggest that NF-kappaB activation in osteoblastic cells constitutes an important pathway in osteoblast-mediated resorptive signalling.
Targets:
 ['Interleukin-1', 'NF-kappa B', 'Nuclear Proteins', 'Parathyroid Hormone', 'Tumor Necrosis Factor-alpha', 'Vitamin D', '1,25-dihydroxyvitamin D']
Preds:
 b'["RNA, Messenger", "Cytokines", "Transcription Factors", "DNA-Binding Proteins", "Tumor Necrosis Factor-alpha", "NF-kappa B", "Membrane Proteins", "Nuclear Proteins"]'
Text:
 Intracellular cAMP regulates cell proliferation as a second messenger of extracellular signals in a number of cell types. We investigated, by pharmacological means, whether an increase in intracellular cAMP levels changes proliferation rates of lactotrophs in primary culture, whether there are interactions between signal transduction pathways of cAMP and the growth factor insulin, and where the dopamine receptor agonist bromocriptine acts in the cAMP pathway to inhibit lactotroph proliferation. Rat anterior pituitary cells, cultured in serum-free medium, were treated with cAMP-increasing agents, followed by 5-bromo-2'-deoxyuridine (BrdU) to label proliferating pituitary cells. BrdU-labeling indices indicative of the proliferation rate of lactotrophs were determined by double immunofluorescence staining for PRL and BrdU. Treatment with forskolin (an adenylate cyclase activator) or (Bu)2cAMP (a membrane-permeable cAMP analog) increased BrdU-labeling indices of lactotrophs in a dose- and incubation time-dependent manner. The cAMP-increasing agents were also effective in increasing BrdU-labeling indices in populations enriched for lactotrophs by differential sedimentation. The stimulatory action of forskolin was observed, regardless of concentrations of insulin that were added in combination with forskolin. Inhibition of the action of endogenous cAMP by H89 or KT5720, a protein kinase A inhibitor, attenuated an increase in BrdU-labeling indices by insulin treatment. On the other hand, the specific mitogen-activated protein kinase inhibitor PD98059, which was effective in blocking the mitogenic action of insulin, markedly suppressed the forskolin-induced increase in BrdU-labeling indices. (Bu)2cAMP antagonized not only inhibition of BrdU labeling indices but also changes in cell shape induced by bromocriptine treatment, although forskolin did not have such an antagonizing effect. These results suggest that: 1) intracellular cAMP plays a stimulatory role in the regulation of lactotroph proliferation; 2) cAMP and insulin/mitogen-activated protein kinase signalings require each other for their mitogenic actions; and 3) the antimitogenic action of bromocriptine is, at least in part, caused by inhibition of cAMP production.
Targets:
 ['Cyclic AMP Response Element-Binding Protein', 'Flavonoids', 'Insulin', 'Colforsin', 'Bromocriptine', 'Bucladesine', 'Cyclic AMP', 'Cyclic AMP-Dependent Protein Kinases', 'Calcium-Calmodulin-Dependent Protein Kinases', '2-(2-amino-3-methoxyphenyl)-4H-1-benzopyran-4-one']
Preds:
 b'["RNA, Messenger", "Calcium", "Cyclic AMP", "Dopamine", "Nerve Tissue Proteins", "Membrane Proteins", "Adenosine Triphosphate", "Protein Kinase C"]'
Text:
 The carbon-14-labeled carbon dioxide that is released by respiration after glucose labeled with carbon-14 is applied to fungal mycelium can be reabsorbed in highly significant amounts by distant mycelium and agar media in the same petri dishes. Atmospheric transfer of carbon-14 must be considered when using labeled organic compounds to study translocation in fungi.
Targets:
 ['Carbon Isotopes', 'Carbon Dioxide']
Preds:
 b'["Water Pollutants, Chemical", "Water", "Culture Media", "Soil", "Soil Pollutants", "Nitrogen", "Carbon", "Sewage"]'
Text:
 Cystic fibrosis (CF) is a consequence of defective recognition of the multimembrane spanning protein cystic fibrosis conductance transmembrane regulator (CFTR) by the protein homeostasis or proteostasis network (PN) (Hutt and Balch (2010). Like many variant proteins triggering misfolding diseases, mutant CFTR has a complex folding and membrane trafficking itinerary that is managed by the PN to maintain proteome balance and this balance is disrupted in human disease. The biological pathways dictating the folding and function of CFTR in health and disease are being studied by numerous investigators, providing a unique opportunity to begin to understand and therapeutically address the role of the PN in disease onset, and its progression during aging. We discuss the general concept that therapeutic management of the emergent properties of the PN to control the energetics of CFTR folding biology may provide significant clinical benefit.
Targets:
 ['CFTR protein, human', 'Cystic Fibrosis Transmembrane Conductance Regulator']
Preds:
 b'["Biomarkers", "Proteins", "MicroRNAs", "Transcription Factors", "Antineoplastic Agents", "DNA-Binding Proteins", "Membrane Proteins", "Biomarkers, Tumor"]'
Text:
 Excessive bleeding and transfusion increase morbidity and mortality in patients receiving coronary artery bypass grafting (CABG), especially in those exposed to antiplatelet agents.
Targets:
 ['Antifibrinolytic Agents', 'Platelet Aggregation Inhibitors', 'Tranexamic Acid', 'Clopidogrel', 'Ticlopidine']
Preds:
 b'["Immunosuppressive Agents", "Adrenal Cortex Hormones", "Biomarkers", "Anticoagulants", "Antineoplastic Agents", "Antihypertensive Agents", "Antibodies, Monoclonal, Humanized", "Antirheumatic Agents"]'
Text:
 Egr-1 is an immediate-early response gene induced transiently and ubiquitously by mitogenic stimuli and also regulated in response to signals that initiate differentiation. The Egr-1 gene product, a nuclear phosphoprotein with three zinc fingers of the Cys2His2 class, binds to the sequence CGCCCCCGC and transactivates a synthetic promoter construct 10-fold in transient-transfection assays. We have analyzed the structure and function of the Egr-1 protein in detail, delineating independent and modular activation, repression, DNA-binding, and nuclear localization activities. Deletion analysis, as well as fusions to the DNA-binding domain of GAL4, indicated that the activation potential of Egr-1 is distributed over an extensive serine/threonine-rich N-terminal domain. In addition, a novel negative regulatory function has been precisely mapped 5' of the zinc fingers: amino acids 281 to 314 are sufficient to confer the ability to repress transcription on a heterologous DNA-binding domain. Specific DNA-binding activity was shown to reside in the three zinc fingers of Egr-1, as predicted by homology to other known DNA-binding proteins. Finally, nuclear localization of Egr-1 is specified by signals in the DNA-binding domain and basic flanking sequences, as determined by subcellular fractionation and indirect immunofluorescence. Basic residues 315 to 330 confer partial nuclear localization on the bacterial protein beta-galactosidase. A bipartite signal consisting of this basic region in conjunction with either the second or third zinc finger, but not the first, suffices to target beta-galactosidase exclusively to the nucleus. Our work shows that Egr-1 is a functionally complex protein and suggests that it may play different roles in the diverse settings in which it is induced.
Targets:
 ['DNA-Binding Proteins', 'Early Growth Response Protein 1', 'Egr1 protein, mouse', 'Immediate-Early Proteins', 'Nuclear Proteins', 'Oligodeoxyribonucleotides', 'Recombinant Fusion Proteins', 'Repressor Proteins', 'Trans-Activators', 'Transcription Factors']
Preds:
 b'["RNA, Messenger", "Transcription Factors", "DNA-Binding Proteins", "Membrane Proteins", "DNA", "Drosophila Proteins", "Recombinant Fusion Proteins", "Recombinant Proteins"]'
Text:
 The study of physical organic chemistry in solution is a mature science, over a century old, but over the last 10 years or so, reversible encapsulation has changed the way researchers view molecular interactions. It is now clear that the behavior of molecules in dilute solution is really quite different from their behavior in capsules. Molecules isolated from bulk media in spaces barely large enough to accommodate them and a few neighbors show new phenomena: their activities resemble those of molecules inside biochemical structures--pockets of enzymes, interiors of chaperones, or the inner space of the ribosome--rather than conventional behavior in solution. In this Account, we recount the behavior of molecules in these small spaces with emphasis on structures and reactivities that have not been, and perhaps cannot be, seen in conventional solution chemistry. The capsules self-assemble through a variety of forces, including hydrogen bonds, metal-ligand interactions, and hydrophobic effects. Their lifetimes range from milliseconds to hours, long enough for NMR spectroscopy to reveal what is going on inside. We describe one particular capsule, the elongated shape of which gives rise to many of the effects and unique phenomena. Molecular guests that are congruent to the space of the host can be tightly packed inside and show reduced mobilities such as rotation and translation within the capsule. These mobilities depend strongly on what else is encapsulated with them. We also relate how asymmetric spaces can be created inside the capsule by using a chiral guest. In contrast to the situation in dilute solution, where rapid exchange of solute partners and free molecular motion average out the steric and magnetic effects of chirality, the long lifetimes of the encounters in the capsules magnify the effects of an asymmetric environment. The capsule remains achiral, but the remaining space is chiral, and coencapsulated molecules respond in an amplified way. We probe the various regions of the capsule with guests of different shape. Primary acetylenes, the narrowest of functional groups, can access the tapered ends of the capsule that exclude functions as small as methyl groups. The shape of the capsule also has consequences for aromatic guests, gently bending some and straightening out others. Flexible structures such as normal alkanes can be compressed to fit within the capsule and conform to its shape. We obtain a measure of the internal pressure caused by the compressed guests by determining its effect on the motion of the capsule's components. These forces can also drive a spring-loaded device under the control of external acids and bases. We show that spacer elements can be added to give self-assembled capsules of increased complexity, with 15 or more molecules spontaneously coming together in the assembly. In addition, we analyze the behavior of gases, including the breakdown of ideal gas behavior, inside these capsules. The versatility of these capsule structures points to possible applications as nanoscale reaction chambers. The exploration of these confined spaces and of the molecules within them continues to open new frontiers.
Targets:
 ['Ligands', 'Metals', 'Organic Chemicals', 'Solutions']
Preds:
 b'["Polymers", "Water", "Proteins", "Biocompatible Materials", "Ligands", "Fluorescent Dyes", "Gold", "Collagen"]'
Text:
 Three indolocarbazole compounds bearing a tripeptide or a lysine group attached to one of the indole nitrogens via a propylamino chain and two rebeccamycin derivatives bearing a lysine residue on the sugar moiety were synthesised with the aim of improving the binding to DNA and the antiproliferative activities. Four tumour cell lines, from murine L1210 leukemia, human HT29 colon carcinoma, A549 non-small cell lung carcinoma and K-562 leukemia, were used to evaluate the cytotoxicity of the drugs. Their effects on the cell cycle of L1210 cells and their antimicrobial properties against two Gram-positive bacteria Bacillus cereus and Streptomyces chartreusis, a Gram-negative bacterium Escherichia coli and a yeast Candida albicans were also investigated.
Targets:
 ['12-(2-O-lysyl-4-O-methyl-glucopyranosyl)-6,7,12,13-tetrahydro(5H)-indolo(2,3-a)-pyrrolo(3,4-c)carbazole-5,7-dione dihydrochloride', 'Amino Acids', 'Aminoglycosides', 'Anti-Bacterial Agents', 'Anti-Infective Agents', 'Antineoplastic Agents', 'Carbazoles', 'Indoles', 'Monosaccharides', 'indolo(3,2-b)carbazole', 'rebeccamycin']
Preds:
 b'["Recombinant Proteins", "Bacterial Proteins", "Antibodies, Monoclonal", "Antineoplastic Agents", "Peptides", "RNA, Messenger", "DNA", "Peptide Fragments"]'
Text:
 A convenient method for isoelectric focusing of intact polymeric IgA and IgM is described. This technique employed composite gels containing 1.0% acrylamide and 0.75% agarose which exhibited minimal electroendosmotic properties. The spectrotypes obtained with mouse IgA myeloma proteins, a human IgA myeloma and rabbit secretory IgA preparations were compared in three gel systems: 5% acrylamide, 0.8% agarose and the composite gel. With respect to resolution of component bands, the composite gel was superior to the other two systems. Hapten binding studies with MOPC-315 IgA and a rabbit secretory IgA anti-DNP antibody indicated that the focused IgA molecules retained their binding site integrity in the composite gel. The pI ranges obtained with microscale sucrose isoelectric focusing and composite gel system showed good correspondence, with the latter exhibiting enhanced resolution. Studies with MOPC-104E IgM revealed improved resolution in the composite gel when compared to the agarose system. Comparison of pI ranges for IgA and IgM immunoglobulins obtained in the present study with those reported previously suggest that IgA spectrotypes are confined to an acidic pI range (3.4--6.4), whereas IgM spectrotypes are not (4.3--8.8).
Targets:
 ['Acrylamides', 'Gels', 'Haptens', 'Immunoglobulin A', 'Immunoglobulin A, Secretory', 'Immunoglobulin M', 'Myeloma Proteins', 'Polysaccharides', 'Sepharose']
Preds:
 b'["Polymers", "Collagen", "Biocompatible Materials", "Polyethylene Glycols", "Antibodies, Monoclonal", "Drug Carriers", "Peptides", "Water"]'
```
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTEzNTIyMjI2MTcsLTE1NTE1MTQzMTEsND
AyODAwMjk5LC0xOTY1ODEwMjc5LC04NDE1Njg1MDksMTAzODIy
MzkzNCwyNDg0MjI3MTEsMTA0NjY4NTM0NF19
-->