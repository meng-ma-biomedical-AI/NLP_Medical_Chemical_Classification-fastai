
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
    
    
#     # TEST MODEL SERVER
#     # Serve model
#     deploy = funcs['model-server'].deploy_step(env={'DATA_CLAS_PATH' : "/User/nlp/run/data_clas.pkl",
#                                                    'MODEL_PATH' : "/User/nlp/run/train_clas_model",
#                                                    f'SERVING_MODEL_{model_endpoint_name}': "/User/nlp/run/train_clas_model",
#                                                    'NUM_PREDS' : num_preds})

#     # Model serving tester
#     tester = funcs['model-server-tester'].as_step(
#         name='model-tester',
#         inputs={'model_endpoint': deploy.outputs['endpoint'],
#                 'model_name' : model_endpoint_name,
#                 'data_size' : data_size,
#                 'data_path' : "/User/nlp/run/data.pkl",
#                 'num_tests' : num_tests})
