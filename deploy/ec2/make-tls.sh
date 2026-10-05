#!/bin/sh
# Make the self-signed Postgres server certificate and key (EC P-256, 10 years).
# Run once on your own machine, then store both in SSM (docs/aws-setup.md);
# write-env.sh copies them onto the instance. Clients pin server.crt with
# sslmode=verify-ca, so it does not matter that the instance IP changes.
# For local development, run it with no argument to write ./tls/.
set -eu
out="${1:-$(dirname "$0")/tls}"
mkdir -p "$out"
umask 077
openssl req -x509 -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
  -days 3650 -subj "/CN=leaf-al-postgres" \
  -keyout "$out/server.key" -out "$out/server.crt" 2>/dev/null
chmod 644 "$out/server.crt"
echo "wrote $out/server.crt and $out/server.key"
