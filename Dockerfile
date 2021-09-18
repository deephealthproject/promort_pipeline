FROM dhealth/pylibs-toolkit:0.10.1-cudnn

# use custom pyeddl with bind_optimizers
RUN \
    cd /usr/local/src \
    && mv pyeddl pyeddl.orig \
    && git clone https://github.com/deephealthproject/pyeddl \
    && cd pyeddl \
    && git checkout bind_optimizers \
    && python3 setup.py install

# Install a few build tools
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y ssh m4 autoconf automake libtool flex pandoc \
    && rm -rf /var/lib/apt/lists/*

# download and compile ucx + gdrcopy
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y libnuma-dev ibverbs-providers perftest strace \
       libibverbs-dev librdmacm-dev binutils-dev gettext rdma-core \
    && rm -rf /var/lib/apt/lists/*

ENV LIBRARY_PATH $LIBRARY_PATH:/usr/local/cuda/lib64/stubs
RUN ldconfig -n /usr/local/cuda/lib64/stub

RUN \
    git clone https://github.com/NVIDIA/gdrcopy.git \
    && cd gdrcopy \
    && make prefix=/usr/local/ lib_install \
    && ldconfig

RUN \
    wget -nv 'https://github.com/openucx/ucx/releases/download/v1.10.1/ucx-1.10.1.tar.gz' \
    && tar xf ucx-1.10.1.tar.gz \
    && cd ucx-1.10.1 \
    && mkdir build \
    && cd build \
    && ../configure --prefix=/usr/local --with-cuda=/usr/local/cuda/ --with-gdrcopy=/usr/local \
    && make -j4 \
    && make install \
    && ldconfig

# download and compile openmpi
RUN \
    wget -nv 'https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.bz2' \
    && tar xf openmpi-4.1.1.tar.bz2 \
    && cd openmpi-4.1.1 \
    && mkdir build \
    && cd build \
    && ../configure --prefix=/usr/local --with-ucx=/usr/local --with-cuda=/usr/local/cuda/ \
    && make all install

RUN ldconfig

# setup ssh and set bash as default shell
RUN \
    bash -c 'echo "PermitRootLogin yes" >> /etc/ssh/sshd_config'

RUN \
    bash -c 'echo -e "* soft memlock unlimited\n* hard memlock unlimited\n" >> /etc/security/limits.conf'

# install some useful tools
RUN \
    export DEBIAN_FRONTEND=noninteractive \
    && apt-get update -y -q \
    && apt-get install -y \
       aptitude \
       bash-completion \
       dnsutils \
       elinks \
       emacs25-nox emacs-goodies-el \
       fish \
       git \
       htop \
       iproute2 \
       iputils-ping \
       ipython3 \
       less \
       mc \
       nload \
       nmon \
       psutils \
       source-highlight \
       tmux \
       vim \
       wget \
       sudo \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -m sgd_mpi \
    && echo "sgd_mpi ALL=(ALL:ALL) NOPASSWD: ALL" | sudo tee /etc/sudoers.d/sgd_mpi

WORKDIR /home/sgd_mpi
COPY . /home/sgd_mpi
RUN chown -R sgd_mpi:sgd_mpi /home/sgd_mpi
USER sgd_mpi

ENTRYPOINT \
    sudo service ssh restart \
    && sleep infinity
