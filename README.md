# BIO-AI course project: Exploring a resource map with a PSO algorithm

## Project structure
The repository is divided into four main folders:
* **data**, containing some maps used as examples
* **factorio**, where to put the binary files for the map generator
* **maps**, containing some code experimenting with maps settings and decoding of strings used by Factorio map generator
* **src**, containing the main source code of the project


## Run the project

The project has been developed using Pycharm IDE. 
If the project is opened using this IDE, it will automatically include some existing run configurations.

### Requirements
 - Python 3.6

It is recommended to use a separate virtualenv for this project.
The project has been developed and tested on our Linux laptops. 
Running under other OSes has not been tested.

If matplotlib creates problems while drawing graphs, try to comment the line in the main file that forces it to use specific backend `matplotlib.use("Qt5Agg")`.

### Dependencies
Pip is used to manage dependencies. You can install them using `pip install -r requirements.txt`

### Configuration

Configuration is done using the file `src/configuration.py`.
The parameters can be configured via environment variables. 
The available variables can be found inside the file.


### Run the main file
Execute the following commands from the main folder of the project, where this `README` is located.

First, we need to set the environment variables used for configuration.
The only required variable is IMAGE_NAME. It is the name of the map image, extension included.

If a variable is not defined, a default value is used, as specified in the `configuration.py` file

The seed (if not specified using env variable SEED) is taken as the current timestamp.

```bash
export IMAGE_NAME=map6.png

python -m src.main
```


### Benchamrks

The benchmarks have been conducted on the map `map6-cropped1.png`

## Map generation

### Requirements

Factorio headless binary. See https://factorio.com/download-headless.

To generate a new map use the command below or directly run the bash script 
**generate.sh** available in the ``maps`` folder:

```bash
./factorio/bin/x64/factorio 
	--generate-map-preview ./data/examples/map_$(date +%Y-%m-%d-%H:%M:%S).png 
	--map-preview-scale 2 
	--map-preview-size 1024 
	--map-gen-settings ./maps/bio-map-gen-settings.json
```

This will create a new map in the `data/example` folder with a name in the format `map_<current_timestamp>.png`.

Parameter `map-preview-scale` manages the scale (1 means 1 meter per pixel).

Other parameters are tunable in the `json` settings file. 
