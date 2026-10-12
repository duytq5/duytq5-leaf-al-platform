#!/bin/sh
# Make the Postgres TLS files (EC P-256, 10 years): a private CA that clients
# pin, and a server certificate signed by it. Run once on your own machine,
# then store them in SSM (docs/aws-setup.md); write-env.sh copies the server
# pair onto the instance. Clients pin ca.crt; the server certificate names
# localhost, so sslmode=verify-full works through the SSM tunnel.
#
# Strict TLS clients (rustls, used by some GUI clients) reject a bare
# self-signed or X.509 v1 certificate, which `openssl req -x509` makes on
# macOS (LibreSSL). Hence the CA + v3 server certificate. The CA key is not
# kept: to change anything, run this again and replace all three parameters.
#
# For local development, run it with no argument to write ./tls/.
set -eu
out="${1:-$(dirname "$0")/tls}"
mkdir -p "$out"
umask 077
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

cat > "$work/ext.cnf" <<'EOF'
[req]
distinguished_name = dn
[dn]
[ca]
basicConstraints = critical, CA:TRUE, pathlen:0
keyUsage = critical, keyCertSign, cRLSign
subjectKeyIdentifier = hash
[server]
basicConstraints = critical, CA:FALSE
keyUsage = critical, digitalSignature
extendedKeyUsage = serverAuth
subjectAltName = DNS:leaf-al-postgres, DNS:localhost, DNS:postgres, IP:127.0.0.1
authorityKeyIdentifier = keyid
subjectKeyIdentifier = hash
EOF

openssl req -x509 -new -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
  -days 3650 -subj "/CN=leaf-al-postgres-ca" -config "$work/ext.cnf" -extensions ca \
  -keyout "$work/ca.key" -out "$out/ca.crt" 2>/dev/null
openssl req -new -newkey ec -pkeyopt ec_paramgen_curve:prime256v1 -nodes \
  -subj "/CN=leaf-al-postgres" -config "$work/ext.cnf" \
  -keyout "$out/server.key" -out "$work/server.csr" 2>/dev/null
openssl x509 -req -in "$work/server.csr" -CA "$out/ca.crt" -CAkey "$work/ca.key" \
  -set_serial "0x$(openssl rand -hex 16)" -days 3650 \
  -extfile "$work/ext.cnf" -extensions server -out "$out/server.crt" 2>/dev/null
chmod 644 "$out/ca.crt" "$out/server.crt"
echo "wrote $out/ca.crt, $out/server.crt and $out/server.key"
