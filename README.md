# Bio-Inspired Artificial Intelligence course project

The repository is divided into three main folders:
* **data** containing some maps used as examples
* **maps** containing some code experimenting with maps settings and de-encoding of strings
* **src** containing different Python scripts (Sci-kit images scripts)


#### Generate a new map
To generate a new map it's possible to use this command or directly run the bash script 
**generate.sh** in the ``maps`` folder:

```bash
./factorio/bin/x64/factorio 
	--generate-map-preview ./data/examples/map_$(date +%Y-%m-%d-%H:%M:%S).png 
	--map-preview-scale 2 
	--map-preview-size 1024 
	--map-gen-settings ./maps/bio-map-gen-settings.json
```

This will create a new map with as name map_(timestamp).png in the ``data/example`` folder.

Parameter ``map-preview-scale`` manage the scale with 1 meaning 1 meter per pixel.

Other generation parameters are in the setting file. 



## How to run the project

```bash

export IMAGE_NAME=map6.png

python -m src.main

```


#### TODO list (30-05-2019)
Problems to be solved
- [x] Add comments to main file (now it is a bit obscure)
- [x] Refactor multiprocessor support ( add a global switch for easy on-off)
- [x] Refactor variator (issue with the last particle position which is the same causing the first individual to be stationary)
- [x] Implement terminator function (a possible criterion may be check if the current position does not change for at least X iterations)
- [ ] Refactor loading cached matrices
- [ ] Add velocity/acceleration to particles
	- [ ] Bound velocity
	- [ ] Dynamic acceleration
- [ ] Optimize resource count evaluation (using memoization)
- [ ] Refactor of fitness_evaluator function (why tan and atan? normalization_factor has always the same factor...)
