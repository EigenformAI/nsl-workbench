FROM python:3.11-slim

ARG MS_AGENT_VERSION=1.6.0
ARG MCP_VERSION=1.27.0
ARG OMEGACONF_VERSION=2.3.0
ARG LOGURU_VERSION=0.7

# Pin the harness and its MCP API dependency. MS-agent 1.6.0 uses
# streamablehttp_client, which was removed in MCP 2.x. These pins do not
# lock the base image or every transitive package.
RUN pip install --no-cache-dir \
    "ms-agent==${MS_AGENT_VERSION}" \
    "mcp==${MCP_VERSION}" \
    "omegaconf>=${OMEGACONF_VERSION}" \
    "loguru>=${LOGURU_VERSION}"

COPY run.py /opt/nsl/run.py
COPY inject_query_callback.py /opt/nsl/inject_query_callback.py

# Stamp the build-context content hash (computed by ensure_harness_image)
# into a Docker label so the next run can detect drift between this image
# and the source files on disk, allowing provisioning to rebuild a stale image.
ARG NSL_BUILD_CONTEXT_SHA=unknown
LABEL nsl.build-context-sha=$NSL_BUILD_CONTEXT_SHA

# ms-agent's logger (utils/logger.py:50) resolves its log path as
# os.path.join(os.getcwd(), 'ms_agent.log') at import time. Without a
# WORKDIR, CWD is `/` which is 0755 and unwritable to the non-root
# user the harness launches the container as (`--user <host uid>`),
# so import of `ms_agent` crashes with PermissionError: '/ms_agent.log'.
# /tmp is 1777 on every Debian-based image — safe for any uid.
WORKDIR /tmp

ENTRYPOINT []
CMD ["python", "/opt/nsl/run.py"]
