# Postgres 16 with the pgvector extension, for the knowledge service's embedding index.
# Same base as before (postgres:16-alpine), so an existing data volume keeps working.
FROM postgres:16-alpine

ARG PGVECTOR_VERSION=0.8.7

# with_llvm=no: the Alpine image has no clang for JIT bitcode. OPTFLAGS="": no -march=native, so the image
# runs on any CPU of the same architecture, not only the one that built it.
RUN apk add --no-cache --virtual .build-deps build-base git \
    && git clone --branch "v${PGVECTOR_VERSION}" --depth 1 https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && make -C /tmp/pgvector with_llvm=no OPTFLAGS="" \
    && make -C /tmp/pgvector with_llvm=no install \
    && rm -rf /tmp/pgvector \
    && apk del .build-deps
