# See here for image contents: https://github.com/microsoft/vscode-dev-containers/tree/v0.166.1/containers/python-3-anaconda/.devcontainer/base.Dockerfile

FROM mcr.microsoft.com/vscode/devcontainers/miniconda:0-3
#FROM mcr.microsoft.com/vscode/devcontainers/anaconda:0-3

# Install additional OS packages.
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
     && apt-get -y install --no-install-recommends make

# Copy environment.yml (if found) to a temp location so we update the environment. Also
# copy "noop.txt" so the COPY instruction does not fail if no environment.yml exists.
COPY environment.yml* .devcontainer/noop.txt /tmp/conda-tmp/
RUN if [ -f "/tmp/conda-tmp/environment.yml" ]; then \
     /opt/conda/bin/conda env create -f /tmp/conda-tmp/environment.yml; \
     /opt/conda/bin/conda clean --all -y; \
     echo "source activate $(head -1 /tmp/conda-tmp/environment.yml | cut -d ' ' -f2)" > ~/.bashrc; \
     rm -rf /tmp/conda-tmp; \
     fi 

ENV PATH /opt/conda/envs/maya-challenge/bin:$PATH

COPY .devcontainer/jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
CMD ["bash", "-c", "jupyter notebook --allow-root --config=/root/.jupyter/jupyter_notebook_config.py"]
