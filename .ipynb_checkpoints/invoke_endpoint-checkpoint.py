#!/usr/bin/python3
# Copyright 2020 Google LLC
# modified by 'replytobishnu@gmail.com'
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [START aiplatform_predict_tabular_classification_sample]
from typing import Dict

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value


def predict_tabular_classification_sample(
    project: str,
    endpoint_id: str,
    instance_dict: Dict,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    # for more info on the instance schema, please use get_model_sample.py
    # and look at the yaml found in instance_schema_uri
    instance = json_format.ParseDict(instance_dict, Value())
    instances = [instance]
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    print("response")
    print(" deployed_model_id:", response.deployed_model_id)
    # See gs://google-cloud-aiplatform/schema/predict/prediction/tabular_classification_1.0.0.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        print(" prediction:", dict(prediction))
    return predictions


# [END aiplatform_predict_tabular_classification_sample]

"""
sample function call below:
predict_tabular_classification_sample(
    project="189737161361",
    endpoint_id="3676841650073632768",
    location="us-central1",
    instances=[{ "feature_column_a": "value", "feature_column_b": "value" ...}, {...}]
)
"""

if __name__ == "__main__":
    import json, sys
    from pathlib import Path 
    import pandas as pd
    csv_path =  Path(sys.argv[1])
    # TODO - If the input is json and not csv #Pass all as string
    df = pd.read_csv(csv_path, nrows=1)
    df = df.astype( {'account_length':'string',
                'number_vmail_messages':'string',
                 'total_day_calls':'string',
                 'number_customer_service_calls':'string',
                'total_day_minutes':'string',
                'total_eve_calls':'string',
                'total_day_charge': 'string',
                'total_eve_minutes':'string',
                'total_eve_charge':'string',
                'total_night_minutes':'string',
                'total_night_calls':'string',
                 'total_night_charge':'string',
                 'total_intl_minutes':'string',
                 'total_intl_calls':'string',
                 'total_intl_charge':'string'
                } )
    dicts = df.to_dict('records')
    
    result = predict_tabular_classification_sample(
    project="189737161361",
    endpoint_id="3676841650073632768",
    location="us-central1",
    instance_dict=dicts[0]
    )
    