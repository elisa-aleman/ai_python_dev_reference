FROM python:3.11.13-bookworm

ENV HOME="/root"
WORKDIR /root/workspace

# Install poetry with pipx
RUN pip install pipx
ENV PATH="$PATH:/root/.local/bin"
RUN pipx install poetry && \
    pipx inject poetry poetry-plugin-export && \
    pipx install toml-cli

CMD ["/bin/bash"]