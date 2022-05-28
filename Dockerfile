FROM python:3.9

ENV PYTHONUNBUFFERED 1
COPY ./requirements.txt /requirements.txt

RUN pip install -r /requirements.txt
RUN python -m spacy download en_core_web_md
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install -y maven && rm -rf /var/lib/apt/lists/*
RUN mvn dependency:copy-dependencies -DoutputDirectory=./jars -f $(python3 -c 'import importlib; import pathlib; print(pathlib.Path(importlib.util.find_spec("sutime").origin).parent / "pom.xml")')
RUN mkdir /app
COPY . /app
WORKDIR /app
