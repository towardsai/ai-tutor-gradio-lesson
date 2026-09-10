# The AI Tutor ships as a Docker Space, so the runtime is pinned here rather
# than inherited from the Gradio SDK's defaults. Python 3.12 is the floor the
# tai-aitutor toolkit requires.
FROM python:3.12-slim

# Spaces run the container as user 1000. The app writes into its working
# directory at startup, where it unzips the vector store, so that directory has
# to belong to that user rather than to root.
RUN useradd -m -u 1000 user && mkdir -p /home/user/app && chown user:user /home/user/app
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Gradio binds 127.0.0.1 by default, which a container never exposes.
ENV GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["python", "app.py"]
