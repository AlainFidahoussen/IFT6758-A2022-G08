## Environment variable

To build and run the dockers, you must define your COMET_API_KEY environment variable first, by:
 - exporting it on the bash prompt: 
```bash
 export COMET_API_KEY=<your key>
```
 - or defining it in a .env file placed on the root directory.

## Docker
The dockers only work on Linux for now (or WSL on Windows).

To build and run the docker serving, the build.sh and run.sh scripts can be used:
```bash
./build.sh
./run.sh
```

To build both the streamlit and serving dockers, the following command, from the root directory, can be used:
```bash
docker compose up
```

The serving log file is available at the address `http://localhost:8890`.

The streamlit service is available at the address `http://localhost:8892`.



