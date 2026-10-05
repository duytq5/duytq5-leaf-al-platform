#!/bin/sh
# Entrypoint wrapper: Postgres refuses a private key that is not owned by its
# own user with mode 0600, so copy the key (written by write-env.sh or
# make-tls.sh, owned by the host user) into place before starting.
set -eu
dir=/var/lib/postgresql/tls
mkdir -p "$dir"
cp /tls/server.crt /tls/server.key "$dir/"
chown -R postgres:postgres "$dir"
chmod 700 "$dir"
chmod 600 "$dir/server.key"
exec docker-entrypoint.sh "$@"
