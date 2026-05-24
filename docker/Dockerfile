FROM nvcr.io/nvidia/pytorch:24.03-py3

# Set timezone and disable interactive prompts
ENV DEBIAN_FRONTEND=noninteractive TZ=UTC

# 1) Install base packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libsm6 \
        libxext6 \
        poppler-utils \
        ca-certificates \
        curl \
        git \
        vim \
        tzdata \
        libgl1 \
        htop \
        ncdu \
        tmux \
        software-properties-common \
        libgl1-mesa-glx \
        tesseract-ocr-osd \
        tesseract-ocr \
        python3.10-venv && \
    ln -fs /usr/share/zoneinfo/$TZ /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# # Install Ookla speedtest
# RUN curl -s https://packagecloud.io/install/repositories/ookla/speedtest-cli/script.deb.sh | bash && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends speedtest

# Verify Python version as root (should print Python 3.11.x)
RUN python3 --version

# Verify Python version as the new user
RUN echo "Python version: $(python3 --version)"

CMD ["/bin/bash"]