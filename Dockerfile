FROM python:3.8-slim-buster
WORKDIR /app
COPY . .
RUN pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple
CMD [ "python", "app.py" ]