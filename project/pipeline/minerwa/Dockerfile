FROM python:3.11-bullseye

ENV PYARROW_WITH_PARQUET=on

RUN apt update -y && \
    apt install -y wget && \
    wget https://apache.jfrog.io/artifactory/arrow/debian/apache-arrow-apt-source-latest-bullseye.deb -O /tmp/arrow.deb && \
    apt install -y /tmp/arrow.deb

RUN apt update -y && \
    apt upgrade -y && \
    apt install -y build-essential cmake cmake-data pkg-config libarrow-dev=8.0.0-1 libarrow-python-dev=8.0.0-1 libthrift-0.13.0 libparquet-dev=8.0.0-1 libparquet-glib800 openjdk-17-jre-headless && \
    apt clean -y

RUN pip install poetry

COPY ./nanodec_requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

COPY vendor /tmp/vendor
RUN pip install /tmp/vendor/vnet_nanodec-1.0.0.tar.gz \
    pip install /tmp/vendor/vnet_nanodec_gee-1.0.0.tar.gz

COPY . /tmp/minerwa
RUN cd /tmp/minerwa && poetry config virtualenvs.create false && poetry install
WORKDIR /tmp/minerwa

ENTRYPOINT ["poetry", "run", "python", "-m", "minerwa.main"]
