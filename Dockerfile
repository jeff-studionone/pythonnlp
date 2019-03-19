FROM python:3.7-stretch

RUN apt-get update && \
    apt-get install -y git

# Get around directory structure quirk of exnteded image
RUN mkdir -p /var/www
# install keras
RUN pip install keras \
    && mkdir -p /var/www/packages \
    && cd /var/www/packages \
    && git clone https://github.com/keras-team/keras.git \
    && pip install --upgrade tensorflow \
    && pip install --upgrade google-api-python-client \
    && pip install --upgrade scikit-learn


RUN cd /var/www/packages/keras \
    && python setup.py install \
    && cd /var/www/ \
    && python -c "import tensorflow as tf; tf.enable_eager_execution(); print(tf.reduce_sum(tf.random_normal([1000, 1000])))"

# pandas
# scikit-learn
# tensorflow
# google-api-python-client

# numpy,
# six,
# h5py,
# keras-applications,
# scipy,
# pyyaml,
# keras-preprocessing,
# keras



WORKDIR /var/www
