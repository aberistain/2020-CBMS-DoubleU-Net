FROM tensorflow/tensorflow

WORKDIR /opt
COPY . /opt

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

CMD python use_trained_model_for_prediction.py  /input-dataset /output-folder
