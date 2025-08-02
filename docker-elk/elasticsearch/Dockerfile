ARG ELASTIC_VERSION

# https://www.docker.elastic.co/
FROM docker.elastic.co/elasticsearch/elasticsearch:${ELASTIC_VERSION:-9.0.4}

# Add your elasticsearch plugins setup here
RUN elasticsearch-plugin install analysis-nori
