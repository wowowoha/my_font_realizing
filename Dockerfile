FROM python:3.8-slim-buster
WORKDIR /app
COPY . . 
RUN pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
CMD ["python3", "wulawual.py"]