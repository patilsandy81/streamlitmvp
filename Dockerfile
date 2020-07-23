FROM python:3.7-slim

EXPOSE 8501

WORKDIR /usr/src/app

# Install pip requirements
ADD requirements.txt .
RUN python -m pip install -r requirements.txt

ENTRYPOINT ["sh", "-c","streamlit run train.py"]
