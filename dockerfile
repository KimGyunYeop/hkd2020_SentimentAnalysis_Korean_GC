# Dockerfile

# without an file extension



# Use an official Python runtime as a parent image

FROM python:3.7-slim



ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV GOOGLE_APPLICATION_CREDENTIALS /service/scripts/application_default_credentials.json

 # Set the working directory to /app

WORKDIR /app



 # Copy the current directory contents into the container at /app

COPY . /app




# Install any needed packages specified in requirements.txt

RUN pip install --trusted-host pypi.python.org -r requirements.txt


RUN pip install pipenv
RUN pipenv install --deploy --system

 # Make port 80 available to the world outside this container

EXPOSE 80



 # Define environment variable

ENV NAME World



 # Run app.py when the container launches

CMD python train.py --result_dir $result_dir --model_mode $model_mode --gpu $gpu