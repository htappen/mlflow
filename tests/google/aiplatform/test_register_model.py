import mlflow
import os
import pytest
import sys
from unittest import mock

import sklearn.datasets as datasets
import sklearn.linear_model as glm

pytestmark = pytest.mark.skipif(
    (sys.version_info < (3, 0)), reason="Tests require Python 3 to run!"
)

# TODO: mock the default project ID setting

class RegisterTestMocks:
    def __init__(self):
        self.mocks = {
            "push_container": mock.patch("docker.models.images.ImageCollection.push"),
            "upload_model": mock.patch("google.cloud.aiplatform.gapic.ModelServiceClient.upload_model"),
            "mlflow_version": mock.patch("mlflow.version.VERSION", "1.13")
        }

    def __getitem__(self, key):
        return self.mocks[key]

    def __enter__(self):
        for key, mock in self.mocks.items():
            self.mocks[key] = mock.__enter__()
        return self

    def __exit__(self, *args):
        for mock in self.mocks.values():
            mock.__exit__(*args)

@pytest.fixture(scope="module")
def model_display_name():
    return "test_mlflow_model"

@pytest.fixture(scope="module")
def sklearn_data():
    iris = datasets.load_iris()
    x = iris.data[:, :2]
    y = iris.target
    return x, y


@pytest.fixture(scope="module")
def sklearn_model(sklearn_data):
    x, y = sklearn_data
    linear_lr = glm.LogisticRegression()
    linear_lr.fit(x, y)
    return linear_lr

@pytest.fixture
def model_path(tmpdir):
    return os.path.join(str(tmpdir), "model")

@pytest.mark.large
def test_register_model_with_absolute_model_path_calls_expected_google_routines(sklearn_model, model_path):
    mlflow.sklearn.save_model(sk_model=sklearn_model, path=model_path)
    with RegisterTestMocks() as mocks:
        mlflow.google.aiplatform.register_model(
            model_path,
            model_display_name
        )

        assert mocks["push_container"].call_count == 1
        assert mocks["upload_model"].call_count == 1
