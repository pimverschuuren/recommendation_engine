FROM python:3.10-slim-buster

WORKDIR /python-docker

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /python-docker

ENV PYTHONPATH /python-docker

CMD ["bash"]
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]