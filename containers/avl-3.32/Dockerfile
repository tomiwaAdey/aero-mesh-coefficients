FROM debian:bookworm-slim

ARG AVL_URL=https://web.mit.edu/drela/Public/web/avl/avl3.32.tgz
ARG AVL_SHA256=dcc0a6ae7ee2f5b3d6a46e730eecab1a1e2a4472a0cd3314bc7fd464f7e091f0

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gfortran \
        libx11-dev \
        make \
    && rm -rf /var/lib/apt/lists/*

RUN curl -fsSLo /tmp/avl.tgz "$AVL_URL" \
    && echo "$AVL_SHA256  /tmp/avl.tgz" | sha256sum -c - \
    && mkdir -p /opt/avl \
    && tar -xzf /tmp/avl.tgz -C /opt/avl --strip-components=1 \
    && cd /opt/avl/plotlib \
    && cp config.make.gfortran config.make \
    && make \
    && cd /opt/avl/eispack \
    && make FC=gfortran FLG=-O2 DP= \
    && cd /opt/avl/bin \
    && cp Makefile_gfortran Makefile \
    && make BINDIR=/usr/local/bin/ PLTLIB='-lX11' avl \
    && rm /tmp/avl.tgz

ENTRYPOINT ["/usr/local/bin/avl"]
