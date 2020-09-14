# Dockerfile

# without an file extension



# Use an official Python runtime as a parent image

FROM python:3.7-slim



ARG result_dir
ARG model_mode
ARG gpu

ENV result_dir ${result_dir}
ENV model_mode ${model_mode}
ENV gpu ${gpu}

RUN echo ${result_dir}.
RUN echo ${model_mode}.
RUN echo ${gpu}.
 # Set the working directory to /app

WORKDIR /app



 # Copy the current directory contents into the container at /app

COPY . /app




# Install any needed packages specified in requirements.txt

RUN pip install --trusted-host pypi.python.org -r requirements.txt



 # Make port 80 available to the world outside this container

EXPOSE 80



 # Define environment variable

ENV NAME World



 # Run app.py when the container launches

CMD python train.py --result_dir $result_dir --model_mode $model_mode --gpu $gpu