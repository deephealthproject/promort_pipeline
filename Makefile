IMG := "eddl-tissue-mask"

.PHONY: docker
docker:
	mkdir -p build-docker
	cp -r python build-docker
	docker $(BUILD_DOCKER_OPTS) build -f docker/Dockerfile -t $(IMG) build-docker

clean:
	rm -r build-docker
