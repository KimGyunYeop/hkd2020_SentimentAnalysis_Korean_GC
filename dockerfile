# Dockerfile

# without an file extension



#  Use an official Python runtime as a parent image

FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel



 # Set the working directory to /app

WORKDIR /app



 # Copy the current directory contents into the container at /app

COPY . /app


RUN pwd

# Install any needed packages specified in requirements.txt

RUN pip install --trusted-host pypi.python.org -r requirements.txt


 # Make port 80 available to the world outside this container

EXPOSE 80



 # Define environment variable
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV NAME World
ENV RESULT_DIR None
ENV MODEL_MODE None
ENV TEST False


CMD python3 -u train.py --result_dir $RESULT_DIR --model_mode $MODEL_MODE --gpu 0