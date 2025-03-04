FROM python:3.13-slim

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

ARG NODE_MAJOR=20

RUN apt-get update \
  # dependencies for building Python packages
  && apt-get install -y build-essential \
  # psycopg2 dependencies
  && apt-get install -y libpq-dev \
  # Translations dependencies
  && apt-get install -y gettext \
  # git is required for pip install from git repo
  && apt-get install -y git ssh \
  # libmagic1 is python-magic's dependency
  && apt-get install -y libmagic1 \
  # nodejs
  && apt-get -yq install curl gnupg ca-certificates \
  && curl -L https://deb.nodesource.com/setup_$NODE_MAJOR.x | bash \
  && apt-get update -yq \
  && apt-get install -yq nodejs \
  # cleaning up unused files
  && apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
  && rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt


COPY ./compose/local/django/entrypoint /entrypoint
RUN sed -i 's/\r$//g' /entrypoint
RUN chmod +x /entrypoint

COPY ./compose/local/django/start /start
RUN sed -i 's/\r$//g' /start
RUN chmod +x /start

WORKDIR /app
COPY . /app

ENTRYPOINT ["/entrypoint"]
