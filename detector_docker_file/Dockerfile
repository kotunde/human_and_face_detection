FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update 
RUN apt-get install -y terminator
RUN apt-get update --fix-missing -y
RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install python3-opencv -y
RUN apt install python3-pip -y
RUN pip3 install tensorflow==2.3.0
RUN pip3 install numpy
RUN pip3 install matplotlib
RUN pip3 install tqdm
RUN pip3 install imutils
RUN apt install htop
RUN apt install lm-sensors
RUN apt install git -y
#RUN pip3 install random
#CMD [ "gimp" ]
#RUN pip3 install extra_keras_datasets


###################################################################################################################################################
#masolas hostrol a container fele
#sudo docker cp ~/Documents/tunde_szrm testtwo:home

#a grafikus programok futasahoz szukseges az X server konfiguracioja
#xhost +

#nevet ad a containernek
#sudo docker run --name det_con -ti --privileged -e DISPLAY=$DISPLAY -v /dev/video0:/dev/video0 -v /tmp/.X11-unix/:/tmp/.X11-unix:ro detector

###################################################################################################################################################
#nem fontos
#sudo docker start -ti --privileged -e DISPLAY=$DISPLAY -v /dev/video0:/dev/video0 -v /tmp/.X11-unix/:/tmp/.X11-unix:ro testtwo
#sudo docker run -ti --privileged -v /dev/video0:/dev/video0 test
#--net=host -e DISPLAY=:0

