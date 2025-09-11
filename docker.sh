docker run -it --name fpp python:3.10-slim /bin/bash
docker cp "C:\Users\34707\Downloads\futures\futures\saved_models\add_close_spread\inference.zip" <docker-id>:/
# enter the container and run:
# export http_proxy=http://172.26.208.1:7890
# export https_proxy=http://172.26.208.1:7890

apt update
apt install git libgomp1 vim zip -y
ssh-keygen -t rsa -C "347073775@qq.com"
cat /root/.ssh/id_rsa.pub
git clone git@github.com:rainyNighti/futures.git

cd futures
git pull
cd ..
cp -r futures/* .
cp inference.py run.py

mkdir inference
unzip inference.zip -d inference

pip install uv
uv pip install -r requirements.txt --system
python run.py