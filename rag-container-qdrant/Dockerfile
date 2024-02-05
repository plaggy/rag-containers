FROM ubuntu:22.04

ARG DOCS_DIR

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget \
    curl \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libffi-dev \
    liblzma-dev \
    git

ENV HOME="/root"
WORKDIR ${HOME}

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN eval "$(pyenv init -)"

RUN pyenv install 3.10
RUN pyenv global 3.10

RUN pip install --upgrade pip
COPY rag-gradio-async/requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /usr/src

COPY rag-gradio-async app
COPY ${DOCS_DIR} docs_dir

COPY embed_and_index.py .
RUN chmod +x embed_and_index.py

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/usr/src/entrypoint.sh"]