FROM debian:bookworm-slim

ARG GMSH_URL=https://gmsh.info/bin/Linux/gmsh-2.7.0-Linux64.tgz
ARG GMSH_SHA256=f2303b791cf2e05efa6850de1eac4c70a7586dd1b19db49df133d2b7291e9a4d

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        fontconfig \
        libgl1 \
        libglu1-mesa \
        libice6 \
        libsm6 \
        libx11-6 \
        libxext6 \
        libxft2 \
        libxinerama1 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/gmsh.tgz "$GMSH_URL" \
    && echo "$GMSH_SHA256  /tmp/gmsh.tgz" | sha256sum -c - \
    && tar -xzf /tmp/gmsh.tgz -C /opt \
    && ln -s /opt/gmsh-2.7.0-Linux/bin/gmsh /usr/local/bin/gmsh \
    && rm /tmp/gmsh.tgz

ENTRYPOINT ["/usr/local/bin/gmsh"]
