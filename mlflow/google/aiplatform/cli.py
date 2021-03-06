"""
CLI for azureml module.
"""
import json

import click

from mlflow.google.aiplatform import register_model as do_register_model
from mlflow.utils import cli_args
from mlflow.utils.annotations import experimental


@click.group("google")
def commands():
    """
    Serve models on Google Cloud AI Platform. **These commands require that MLflow be installed with Python 3.**

    To serve a model associated with a run on a tracking server, set the MLFLOW_TRACKING_URI
    environment variable to the URL of the desired server.
    """
    pass


@commands.command(
    short_help=(
        "Build a container and register an MLflow model with"
        " Cloud AI Platform for deployment."
    )
)
@cli_args.MODEL_URI
@click.option(
    "--display-name",
    "-n",
    required=True,
    help="Name of the model when it's registered on Cloud AI Platform",
)
@click.option(
    "--model-options",
    "-o",
    default=None,
    help="JSON string of other attributes of the Cloud AI Platform Model object, like labels and schema",
)
@click.option(
    "--project",
    "-p",
    default=None,
    help="The Google Cloud Platform project in which to build image and register the model,"
         " as a string. Uses the default project from gcloud config set <PROJECT_ID> if not set"
)
@click.option(
    "--destination-image-uri",
    "-t",
    default=None,
    help="The name of the container image that will be built with your model inside of it."
         " Should be of the format ``gcr.io/<REPO>/<IMAGE>:<TAG>``. Defaults to"
         " gcr.io/<DEFAULT PROJECT>/<MODEL NAME>/<LATEST>"
)
@click.option(
    "--location",
    "-l",
    default="us-central1",
    help="The GCP region that your model will be created in. Defaults to us-central1"
)
@click.option(
    "--wait-timeout",
    "-w",
    default=1800,
    help="How long to wait for model deployment to complete. Defaults to 30 minutes."
)
@click.option(
    "--mlflow-source-dir",
    default=None,
    help="Optionally, specify a dir to install MLFlow from instead of PyPI"
)

@experimental
def register_model(
    model_uri: str,
    display_name: str,
    destination_image_uri: str,
    mlflow_source_dir: str,
    model_options: dict,
    project: str,
    location: str,
    wait_timeout: int
):
    """
    Build a container image included your model that's ready to serve on
    Google Cloud AI Platform. Push the container image and register a model.

    After you run this step, you should use gcloud to create an endpoint and deploy the
    model behind it.

    See https://cloud.google.com/ai-platform-unified/docs/predictions/deploy-model-api
    """
    model_cfg = None
    if model_options is not None:
        model_cfg = json.loads(model_options)
    do_register_model(
        model_uri=model_uri,
        display_name=display_name,
        destination_image_uri=destination_image_uri,
        mlflow_source_dir=mlflow_source_dir,
        model_options=model_cfg,
        project=project,
        location=location,
        synchronous=True,
        wait_timeout=wait_timeout
    )
