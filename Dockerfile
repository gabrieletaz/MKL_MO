FROM pytorch/pytorch:latest


ENV TZ=Europe/Minsk
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get upgrade -y && pip3 install --upgrade pip
RUN apt-get install git-all -y
RUN apt-get install tmux -y && echo "set -g mouse on" > ~/.tmux.conf

WORKDIR /supertml_workdir

COPY ./requirements.txt .
RUN pip install -r requirements.txt