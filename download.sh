#!/bin/bash

python3 -m pip install -r requirements.txt


for i in 1243542914 1245044476 1246944446; do
    json=${i}.json
    if [ ! -f "$json" ]; then
        chat_downloader -o $json https://www.twitch.tv/videos/${i}
    fi
done

# ./main.py *.json
