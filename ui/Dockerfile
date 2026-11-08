# RAHBAR ui: nginx serving the chat page, non-root, read-only root filesystem.
FROM nginxinc/nginx-unprivileged:1.27.3-alpine

USER root
RUN apk upgrade --no-cache
COPY nginx.conf /etc/nginx/nginx.conf
COPY index.html /usr/share/nginx/html/index.html
# nginx-unprivileged's user; writes only under /tmp (an emptyDir in the Deployment)
USER 101
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
