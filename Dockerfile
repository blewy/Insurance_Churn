FROM python:3.7.5-slim-buster

RUN mkdir -p /opt/app
COPY requirements.txt /opt/app/requirements.txt
RUN pip install --upgrade pip

# ensure we can run the make commands
RUN apt-get update -y && \
 	apt-get install -y make && \
 	apt-get install -y libffi-dev gcc && \
 	# for swagger
 	apt-get install -y curl

RUN pip install -r /opt/app/requirements.txt
COPY docker/Makefile /opt/app/Makefile
COPY docker/docker-compose.yaml /opt/app/docker-compose.yaml

COPY ml_api/api /opt/app/api
COPY src /opt/app/src

COPY ml_api/run.py /opt/app/run.py

WORKDIR /opt/app
