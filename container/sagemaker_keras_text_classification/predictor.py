# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#   Licensed under the Apache License, Version 2.0 (the "License").
#   You may not use this file except in compliance with the License.
#   A copy of the License is located at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   or in the "license" file accompanying this file. This file is distributed
#   on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#   express or implied. See the License for the specific language governing
#   permissions and limitations under the License.
#
# This is the file that implements a flask server to do inferences. It's the file that you will modify to
# implement the scoring for your own algorithm.

from __future__ import print_function

import os
import json
import pickle
import StringIO
import sys
import signal
import traceback
import json
import flask

import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 100
label_index = {'Business':0,'Science & Technology':1,'Entertainment':2,'Health & Medicine':3}
prefix = '/opt/ml/'
model_path = os.path.join(prefix, 'model')
with open(os.path.join(model_path,'tokenizer.pickle'), 'rb') as handle:
    tokenizer = pickle.load(handle)

def get_class_label(prediction):
  for key, value in label_index.iteritems():
    if value == prediction[0]:
      return key

# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded

    @classmethod
    def get_model(cls):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            #with open(os.path.join(model_path, 'decision-tree-model.pkl'), 'r') as inp:
            cls.model = tf.keras.models.load_model(os.path.join(model_path, 'news_breaker.h5'))
            cls.model._make_predict_function()
        return cls.model

    @classmethod
    def predict(cls, input):
        """For the input, do the predictions and return them.

        Args:
            input (a single news headline): The data on which to do the predictions. """
        clf = cls.get_model()
        seq = tokenizer.texts_to_sequences([input])
        d = pad_sequences(seq, maxlen=MAX_LEN)
        prediction = clf.predict_classes(np.array(d))
        return(get_class_label(prediction))

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = ScoringService.get_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single news headline. """
    data = None

    # Convert from CSV to pandas
    if flask.request.content_type == 'application/json':
        request_json = flask.request.get_json()
        input = request_json.get('input')
    else:
        return flask.Response(response='This predictor only supports JSON format', status=415, mimetype='text/plain')

    #
    # # Do the prediction
    prediction = {
        "result": ScoringService.predict(input)
    }
    return flask.Response(response=json.dumps(prediction), status=200, mimetype='application/json')
