ARG ELASTIC_VERSION

FROM docker.elastic.co/elasticsearch/elasticsearch:${ELASTIC_VERSION:-9.1.0}

RUN elasticsearch-plugin install analysis-nori
