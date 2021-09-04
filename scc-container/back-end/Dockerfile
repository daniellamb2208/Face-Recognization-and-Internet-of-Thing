FROM python:3.7

COPY requirements.txt requirements.txt

RUN apt-get update -y && \
	apt-get install -y python3 python3-pip libgl1-mesa-glx

RUN pip3 install -r requirements.txt

COPY . /app

WORKDIR /app

ENTRYPOINT /app/entry.sh
