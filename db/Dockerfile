FROM postgres:15.2

# buildに必要な依存関係を入れる
RUN apt-get update && \
  apt-get install -y git make gcc postgresql-server-dev-15

# pgvectorをbuildしてinstall
RUN git clone --branch v0.4.4 https://github.com/pgvector/pgvector.git && \
  cd pgvector && \
  make && \
  make install && \
  cd ../ && rm -rf pgvector