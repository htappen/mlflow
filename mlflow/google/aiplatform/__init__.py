"""
The ``mlflow.google`` module provides an API for deploying MLflow models to
Google Cloud AI Platforom.
"""
import docker
import google.auth
import logging

from google.cloud.aiplatform.gapic import ModelServiceClient
from mlflow.models.cli import _get_flavor_backend
from mlflow.utils.annotations import experimental
from mlflow.pyfunc.scoring_server import FORCE_TF_SERVING_OUTPUT 


_logger = logging.getLogger(__name__)

API_ENDPOINT_TEMPLATE = "{location}-aiplatform.googleapis.com"
MODEL_PARENT_TEMPLATE = "projects/{project}/locations/{location}"
IMAGE_URI_TEMPLATE = "gcr.io/{project}/{image_name}"

@experimental
def register_model(
    model_uri: str,
    display_name: str,
    destination_image_uri: str=None,
    mlflow_source_dir: str=None,
    model_options: dict=None,
    project: str=None,
    location: str="us-central1",
    synchronous: bool=True,
    wait_timeout: int = 1800,
):
    """
    Builds a container and register an MLflow model with Google Cloud AI platform.
    The resulting image can be deployed as a web service to Cloud AI Platform Prediction endpoints.

    The resulting container image will contain a webserver that processes model queries.
    For information about the input data formats accepted by this webserver, see the
    :ref:`MLflow deployment tools documentation <google_deployment>`.

    :param model_uri: The location, in URI format, of the MLflow model used to build the container image.
    For example:

                      - ``/Users/me/path/to/local/model``
                      - ``relative/path/to/local/model``
                      - ``gs://my_bucket/path/to/model``
                      - ``runs:/<mlflow_run_id>/run-relative/path/to/model``
                      - ``models:/<model_name>/<model_version>``
                      - ``models:/<model_name>/<stage>``

                      For more information about supported URI schemes, see
                      `Referencing Artifacts <https://www.mlflow.org/docs/latest/concepts.html#
                      artifact-locations>`_.

    :param display_name: The name of the model when it's registered on Cloud AI Platform
    :param destination_image_uri: The name of the container image that will be built with your model inside of it.
                                  Should be of the format ``gcr.io/<REPO>/<IMAGE>:<TAG>``. Defaults to
                                  gcr.io/<DEFAULT PROJECT>/<display_name>:<LATEST>
    :param mlflow_source_dir: If set, installs MLFlow from this directory instead of PyPI
    :param model_options: A dict of other attributes of the Cloud AI Platform Model object, like labels and schema
    :param project: The Google Cloud Platform project in which to build image and register the model,
                    as a string. Uses the default project from gcloud config set <PROJECT_ID> if not set
    :param location: The GCP region that your model will be created in. Defaults to us-central1
    :param synchronous: If ``True``, this method blocks until the image creation procedure
                        terminates before returning. If ``False``, the method returns immediately,
                        but the returned image will not be available until the asynchronous
                        creation process completes.
    :param wait_timeout: How long to wait for model deployment to complete, if synchronous is ``True``.
                         Defaults to 30 minutes.

    :return: The response of the Google Cloud AI Platform API for registering the model

    .. code-block:: python
        :caption: Example

        from mlflow.google.aiplatform import register_model
        from google.cloud.aiplatform.gapic import EndpointServiceClient

        # Use MLFlow to register the model on Cloud AI Platform
        model_uri = "models:/mymodel/mymodelversion" # Change to a path to your model
        display_name = "my_mlflow_model" # Change to a display name to reference on Google Cloud
        endpoint_name = "my_serving_endpoint" # Change to a display name to send predictions to

        model_info = register_model(model_uri, display_name)

        # Create an endpoint...
        # See https://github.com/googleapis/python-aiplatform/blob/master/samples/snippets/create_endpoint_sample.py
        client_options = {"api_endpoint": "us-central1-aiplatform.googleapis.com"}
        _, project = google.auth.default()
        cloud_model_id = model_info.name

        client = EndpointServiceClient(client_options=client_options)
        response = endpoint_client.create_endpoint(
            parent=f"projects/{project}/locations/us-central1", 
            endpoint={"display_name": endpoint_name}
        )
        cloud_endpoint_id = response.result(timeout=timeout).name

        # ...then deploy the model
        # See https://github.com/googleapis/python-aiplatform/blob/master/samples/snippets/deploy_model_custom_trained_model_sample.py
        deployed_model = {
            "model": cloud_model_id,
            "display_name": display_name
            "dedicated_resources": {
                "min_replica_count": 1,
                "machine_spec": { "machine_type": "n1-standard-2" }
            }
        }
        traffic_split = {"0": 100}
        endpoint = client.endpoint_path(
            project=project, location='us-central1, endpoint=cloud_endpoint_id
        )
        response = client.deploy_model(
            endpoint=endpoint, deployed_model=deployed_model, traffic_split=traffic_split
        )

    """
    if not project:
        try:
            _, project = google.auth.default()
            _logger.info(
                "Project not set. Using %s as project",
                project
                )
        except google.auth.exceptions.DefaultCredentialsError as e:
            raise ValueError(
                "You must either pass a project ID in or set a default project"
                " (e.g. using gcloud config set project <PROJECT ID>. Default credentials"
                " not found: {}".format(e.message)
            ) from e

    if not destination_image_uri:
        destination_image_uri = IMAGE_URI_TEMPLATE.format(
            project=project,
            image_name=display_name
        )
        _logger.info(
            "Destination image URI not set. Building and uploading image to %s",
             destination_image_uri
            )

    # TODO: allow passing in MLFlow home for development simplicity
    _build_serving_image(
        model_uri,
        destination_image_uri,
        mlflow_source_dir
    )

    operation = _upload_model(
        destination_image_uri,
        display_name,
        project,
        location,
        model_options
    )
    if synchronous:
        upload_model_response = operation.result(timeout=wait_timeout)
    return upload_model_response

def _build_serving_image(
    model_uri: str,
    destination_image_uri: str,
    mlflow_source_dir: str=None
):
    _logger.info("Building image")
    flavor_backend = _get_flavor_backend(model_uri)
    flavor_backend.build_image(
        model_uri,
        destination_image_uri,
        install_mlflow=mlflow_source_dir is not None,
        mlflow_home=mlflow_source_dir
    )
    _logger.info("Uploading image to Google Container Registry")

    client = docker.from_env()
    result = client.images.push(
        destination_image_uri,
        stream=True,
        decode=True
    )
    for line in result:
        # Docker client doesn't catch auth errors, so we have to do it
        # ourselves. See https://github.com/docker/docker-py/issues/1772
        if 'errorDetail' in line:
            raise docker.errors.APIError(
                line['errorDetail']['message']
            )
        if 'status' in line:
            print(line['status'])



def _upload_model(
    image_uri: str,
    display_name: str,
    project: str,
    location: str,
    model_options: dict
    ):

    model_cfg = {
        "display_name": display_name,
        "container_spec": {
            "image_uri": image_uri,
            "ports": [
                { "container_port": 8080 }
            ],
            "env": [
                { 
                    "name": FORCE_TF_SERVING_OUTPUT,
                    "value": "True"
                }
            ],
            "predict_route": "/invocations",
            "health_route": "/ping",
        }
    }
    if model_options:
        model_cfg.update(model_options)

    api_endpoint = API_ENDPOINT_TEMPLATE.format(location=location)
    client_options = {"api_endpoint": api_endpoint}

    model_client = ModelServiceClient(client_options=client_options)
    model_parent = MODEL_PARENT_TEMPLATE.format(
        project=project,
        location=location
    )
    _logger.info(
        "Uploading config to Google Cloud AI Platform: %s/models/%s",
        model_parent,
        display_name
    )
    response = model_client.upload_model(parent=model_parent, model=model_cfg)
    return response