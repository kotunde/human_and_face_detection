# Human and face detection

This is a university project. It consists of two parts: human detection and face detection.
Documentation: https://docs.google.com/document/d/1X2I4zvU2znlGQrUZwVTkfUP7diPsdmh7K8VVfTlZzvA/edit?usp=sharing


## How to run locally
clone repository

 ```
 git clone https://github.com/kotunde/human_and_face_detection.git
 ```
 
create python3 virtual env
```
python3 -m venv env
. env/bin/activate
```

install requirements
```
pip install -r requirements.txt
```

launch



## How to run in docker container
- install docker [see here](https://docs.docker.com/engine/install/ubuntu/#installation-methods)

- pull docker image 
```
docker pull kovacstunde/detector
```
- create a container from docker image. Clone repository (see first step from "Run locally")
