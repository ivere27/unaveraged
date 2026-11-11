FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install system dependencies, Python 3.10, Make, and Chromium headless dependencies + CJK fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    make \
    python3 \
    python3-venv \
    python3-pip \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    fonts-noto-cjk \
    fonts-liberation \
    xdg-utils \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 22.x LTS
RUN curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create builder user matching host UID/GID (1000:1000)
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} builder && \
    useradd -u ${USER_ID} -g ${GROUP_ID} -m -s /bin/bash builder

USER builder
WORKDIR /workspace

CMD ["make"]
