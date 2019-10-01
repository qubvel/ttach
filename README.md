## Run tests

```bash
docker build -f Dockerfile.dev -t ttach:dev . && docker run --rm ttach:dev pytest -p no:cacheprovider
```