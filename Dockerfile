FROM python:3.7-slim

EXPOSE 8501

WORKDIR /app
ADD . /app

# Install pip requirements

ADD requirements.txt .
RUN python -m pip install -r requirements.txt

ENTRYPOINT [ "streamlit", "run"]
CMD ["app.py"]