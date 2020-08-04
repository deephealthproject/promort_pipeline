# promort_pipeline


# Docker
For building docker, run 

```
make docker
```

You can pass argument using env variable BUILD__DOCKER__OPTS
For example, you can run docker on a remote host setting:
  
```
  export BUILD_DOCKER_OPTS="-H remote.docker.uri"
  make docker
```
