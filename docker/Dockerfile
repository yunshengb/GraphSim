FROM nvcr.io/nvidia/tensorflow:18.05-py3


RUN apt-get update && apt-get install --no-install-recommends -y \
        git \
        python3-tk \
        graphviz \
        libgraphviz-dev \
        pkg-config \
        libboost-all-dev \
  && rm -rf /var/lib/apt/lists/*

RUN pip3 install numpy \
  scipy \
  networkx==1.10 \
  scikit-learn \
  beautifulsoup4 \
  matplotlib \
  lxml \
  seaborn \
  colour \
  pytz \
  pygraphviz \
  klepto \
  requests


# Installs Java.
ENV JAVA_VER 8
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle
RUN echo 'deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main' >> /etc/apt/sources.list && \
    echo 'deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main' >> /etc/apt/sources.list && \
    apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C2518248EEA14886 && \
    apt-get update && \
    echo oracle-java${JAVA_VER}-installer shared/accepted-oracle-license-v1-1 select true | /usr/bin/debconf-set-selections && \
    apt-get install -y --force-yes --no-install-recommends oracle-java${JAVA_VER}-installer oracle-java${JAVA_VER}-set-default && \
    apt-get clean && \
    rm -rf /var/cache/oracle-jdk${JAVA_VER}-installer
RUN update-java-alternatives -s java-8-oracle
RUN echo "export JAVA_HOME=/usr/lib/jvm/java-8-oracle" >> ~/.bashrc


WORKDIR /workspace







# rsync -avzh ~/Documents/GraphEmbedding <>:/home/<>/
# docker build . -t <>_graphembedding
# nvidia-docker run -v <>:/workspace -it <>_graphembedding bash
# pip install klepto requests
# cd model/Siamese && python run.py
# cd model/Siamese && ./baseline_tester.py
