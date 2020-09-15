# Dockerfile

# without an file extension



# Use an official Python runtime as a parent image

FROM python:3.7-slim



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
ENV RESULT_DIR=default_token
ENV MODEL_MODE=default_token



 # Run app.py when the container launches

CMD ["sh" ,"train.sh", "$RESULT_DIR", "$MODEL_MODE"]