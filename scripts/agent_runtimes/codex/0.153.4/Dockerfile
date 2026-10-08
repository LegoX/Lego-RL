# Mounted runtime image for the Codex CLI.
#
# Stage 1 pulls the npm package purely to get at the vendored Rust binary;
# nothing from node survives into the final image.
ARG CODEX_VERSION=0.153.4

FROM node:22-alpine AS fetch
ARG CODEX_VERSION
RUN npm install -g --prefix /opt/npm "@openai/codex@${CODEX_VERSION}"
RUN set -eux; \
    vendor_root=/opt/npm/lib/node_modules/@openai/codex/node_modules/@openai/codex-linux-x64/vendor; \
    target="$(find "$vendor_root" -maxdepth 1 -type d -name 'x86_64-*-linux-musl' | head -1)"; \
    test -n "$target"; \
    mkdir -p /out; \
    cp -a "$target"/. /out/; \
    test -x /out/bin/codex; \
    # Staticness is asserted host-side by build-runtime-image.sh: alpine's musl
    # ldd does not use glibc's "statically linked" wording, and running the
    # binary in this stage proves nothing because musl is present here.
    /out/bin/codex --version

FROM scratch
COPY --from=fetch /out/ /opt/custom-agent-runtime/codex/
COPY --chmod=0755 runtime-env.sh /opt/custom-agent-runtime/codex/runtime-env.sh
