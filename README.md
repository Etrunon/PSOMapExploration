# Bio-Inspired Artificial Intelligence course project

## Project structure
The repository is divided into three main folders:
* **data** containing some maps used as examples
* **factorio** where to put the binary files for the map generator
* **maps** containing some code experimenting with maps settings and de-encoding of strings used by Factorio map generator
* **src** containing the main source code of the project


## Run the project

The project has been developed using Pycharm IDE. 
If the project is opened using this IDE, it will open automatically pick up the saved run configurations.

### Requirements
 - Python 3.6

It is recommended to use a separate virtualenv for this project.
The project has been developed and tested on our Ubuntu laptops. 
Running under other OSes has not been tested.

If matplotlib creates problems while drawing graphs, try commenting the line in the main file that forces it to use specific backend `matplotlib.use("Qt5Agg")`

### Dependencies
Pip is used to manage dependencies

### Configuration

Configuration is done in the file `src/configuration.py`. The parameters can be configured via environment variables


### Run the main file

`pip install -r requirements.txt`

```bash
export IMAGE_NAME=map6.png

python -m src.main
```

## Map generation

### Requirements

Factorio headless binary. See https://factorio.com/download-headless

To generate a new map it's possible to use this command or directly run the bash script 
**generate.sh** available in the ``maps`` folder:

```bash
./factorio/bin/x64/factorio 
	--generate-map-preview ./data/examples/map_$(date +%Y-%m-%d-%H:%M:%S).png 
	--map-preview-scale 2 
	--map-preview-size 1024 
	--map-gen-settings ./maps/bio-map-gen-settings.json
```

This will create a new map with as name `map_<current_timestamp>.png` in the `data/example` folder.

Parameter `map-preview-scale` manage the scale with 1 meaning 1 meter per pixel.

Other generation parameters are tunable in the `json` settings file. 
