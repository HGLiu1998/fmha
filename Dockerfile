# Start with a base image that has the necessary tools.
# Ubuntu 24.04 is a good choice since your rocprof-trace-decoder is for that version.
# Obtain the required ROCm version from https://hub.docker.com/r/rocm/pytorch/tags
FROM rocm/pytorch:latest
#FROM rocm/pytorch:rocm7.1_ubuntu22.04_py3.11_pytorch_release_2.9.1
# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
# Update the package list and install the required dependencies
# The --no-install-recommends flag keeps the image size down
RUN apt-get update                             && \
    apt-get install -y --no-install-recommends    \
        git cmake wget dpkg vim tig net-tools     \
        libdw-dev libsqlite3-dev  ninja-build  && \
    pip install --upgrade pip                  && \
    pip install ninja                          && \
    rm -rf /var/lib/apt/lists/*
# Set the working directory for the build process
WORKDIR /tmp
ENV GIT_SSL_NO_VERIFY=1
# --- Build and install rocprofiler-sdk ---
#RUN git clone https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source && \
#    cd rocprofiler-sdk-source && mkdir build && cd build && \
#    cmake                                    \
#        -B rocprofiler-sdk-build             \
#        -DCMAKE_INSTALL_PREFIX=/opt/rocm     \
#        -DCMAKE_PREFIX_PATH=/opt/rocm        \
#        rocprofiler-sdk-source ..         && \
#    cmake --build rocprofiler-sdk-build --target all --parallel $(nproc) && \
#    cmake --build rocprofiler-sdk-build --target install                 && \
#    rm -rf rocprofiler-sdk-source rocprofiler-sdk-build
# --- Install rocprof-trace-decoder ---
RUN wget "https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.4/rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux.deb" && \
    dpkg -i rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux.deb && \
    rm rocprof-trace-decoder-ubuntu-22.04-0.1.4-Linux.deb
# --- Build and install rocprof-compute ---
RUN pip install pytz python-dateutil plotly typing_extensions \
                flask importlib_metadata plotext textual      \
                textual_plotext textual-fspicker
RUN git clone https://github.com/ROCm/rocprofiler-compute.git && \
    cd rocprofiler-compute && \
    git checkout f0fad19e8b4f5681ecb56e4ebd9e3e8f82741f36 && \
    export INSTALL_PATH=/opt/rocprof-compute && \
    mkdir -p ${INSTALL_PATH}/python-libs && \
    python3 -m pip install --upgrade -t ${INSTALL_PATH}/python-libs --no-cache-dir -r requirements.txt && \
    mkdir -p build && cd build && \
    cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PATH} \
          -DPYTHON_DEPS=${INSTALL_PATH}/python-libs ../ && \
    make -j$(nproc) install && \
    pip install -r ${INSTALL_PATH}/libexec/rocprofiler-compute/requirements.txt \
        --no-dependencies && \
    rm -rf /tmp/rocprofiler-compute
ENV ROCPROF=rocprofv3
# The container is now ready to use with the installed software.
CMD ["/bin/bash"]
