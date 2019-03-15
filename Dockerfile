FROM python:3.7-stretch

RUN apt-get update && \
    apt-get install -y git

# Get around directory structure quirk of exnteded image
RUN mkdir -p /var/www

WORKDIR /var/www
