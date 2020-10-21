FROM nvidia/cuda:10.2-devel-ubuntu18.04

RUN apt-get update && apt-get install -y \
    build-essential \
	git \
	ninja-build \
	software-properties-common \
	pkg-config \
	unzip \
    wget \
    libgl1-mesa-glx \
    libgl1 \
    zsh

ARG PYTHON=python3.8
ENV LANG C.UTF-8

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y \
    ${PYTHON} \
    python3-distutils \
    python3-apt \
    python3-dev \
    ${PYTHON}-dev

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN ${PYTHON} get-pip.py
RUN ln -sf /usr/bin/${PYTHON} /usr/local/bin/python3
RUN ln -sf /usr/local/bin/pip /usr/local/bin/pip3

RUN pip3 --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

ARG DEBIAN_FRONTEND=noninteractive
#ARG UNAME=<your name>
#ARG UID=<your user id>
#ARG GID=<your group id>

ARG BASE=/app
RUN mkdir ${BASE}
RUN mkdir /data

COPY stylegan_code_finder/requirements.txt ${BASE}/requirements.txt
COPY training_tools ${BASE}/training_tools
RUN cd ${BASE}/training_tools && pip3 install .

WORKDIR ${BASE}
RUN pip3 install -r requirements.txt

RUN groupadd -g $GID -o $UNAME
RUN useradd -m -u $UID -g $GID -o -s /bin/zsh $UNAME

USER $UNAME
RUN git clone https://github.com/robbyrussell/oh-my-zsh.git ~/.oh-my-zsh
RUN cp ~/.oh-my-zsh/templates/zshrc.zsh-template ~/.zshrc

CMD ["/bin/zsh"]
