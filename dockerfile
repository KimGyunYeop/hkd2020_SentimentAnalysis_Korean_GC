# Dockerfile

# without an file extension



# Use an official Python runtime as a parent image

FROM python:3.7-slim



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

ENV NAME World
ENV RESULT_DIR None
ENV MODEL_MODE None
ENV TEST False


CMD if $TEST; then python3 train.py --result_dir $RESULT_DIR --model_mode $MODEL_MODE --gpu 0; else python3 test.py --result_dir $RESULT_DIR --model_mode $MODEL_MODE --gpu 0; fi