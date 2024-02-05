#!/bin/bash

until $(curl --output /dev/null --head --fail $EMBED_URL); do
    printf '.'
    sleep 3
done &&

python embed_and_index.py &&
gradio app/app.py