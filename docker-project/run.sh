#!/bin/bash
docker run -it -d -p5000:5000 -e COMET_API_KEY=$COMET_API_KEY ift6758-serving