FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libvulkan-dev \
    mesa-vulkan-drivers \
    vulkan-tools \
    libopencv-dev \
    libglm-dev \
    libshaderc-dev \
    glslang-tools \
    clang-format \
    clang-tidy \
    libocct-foundation-dev \
    libocct-modeling-data-dev \
    libocct-modeling-algorithms-dev \
    libocct-data-exchange-dev \
    nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src

COPY . /src

RUN mkdir -p /opt/maskgen-build && cd /opt/maskgen-build && cmake /src -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && make -j$(nproc) && make install && ldconfig

WORKDIR /workspace

ENTRYPOINT ["maskgen_cli"]
