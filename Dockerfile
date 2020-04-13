FROM python:3.7

RUN apt-get update

RUN mkdir -p /code/bopflow
COPY . /code/bopflow

RUN pip install /code/bopflow
