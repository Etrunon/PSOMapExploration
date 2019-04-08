#!/bin/bash
./factorio/bin/x64/factorio \
	--generate-map-preview ./data/examples/map_$(date +%Y-%m-%d-%H:%M:%S).png \
	--map-preview-scale 2 \
	--map-preview-size 1024 \
	--map-gen-settings ./maps/bio-map-gen-settings.json
