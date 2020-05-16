FROM python:3

WORKDIR /Users/sruthi/Desktop/DSC180BAll/DSC180B

COPY . /DSC180B 

RUN pip install --no-cache-dir -r /Users/sruthi/Desktop/DSC180BAll/DSC180B/requirements.txt

CMD ["python", "./run.py"]