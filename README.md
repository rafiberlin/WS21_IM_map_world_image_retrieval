# A MapWorld Avatar Game For Image Retrieval (WS21 IM)

# Repository Structure
This project is the Avatar Game. It has the following structure:

    ├── /avatar_sgg                          # Game basis
    │   ├── /captioning                      # Code for captioning model used for sentence similarity
    │   ├── /config                          # General configuration of the game
    │   ├── /dataset                         # Contain utility tools to load ADE20K or Visual Genome Data for evaluation 
    │   ├── /image_retrieval                 # Code for the sentence to graph model and evaluation tools
    │   ├── /mapworld                        # The map with the images
    │   ├── /notebooks                       # Some visualizations of results and model output
    │   ├── /resources                       # JSON for the layout and the images
    │   ├── /scripts                         # To initialize the game
    │   ├── game.py                          # Start the game
    │   ├── game_avatar.py                   # Start a dummy avatar
    │   ├── game_avatar_abstract.py          # Base avatar for Image Retrieval
    |   ├── game_avatar_baseline.py          # Avatar performing Image Retrieval on sentence similarity
    |   ├── game_avatar_graph.py             # Avatar performing Image Retrieval with sentence to graph
    │   ├── game_avatar_slurk.py             # Start avatar in slurk
    │   ├── game_master_slurk.py             # Start master in slurk
    │   └── game_master_standalone.py        # Start master
    ├── /results                             # Text output of the different metrics displayed in the final report
    ├── /tests                               # Some MapWorld tests
    └── /setup.py                            # To install the game
    

# Pre-requisite

You can play the game with 2 avatars, the baseline avatar and the Sentence-To-Graph avatar.

The model for the Sentence-To-Graph avatar can be downloaded under:
`https://drive.google.com/file/d/1ViWIsK5W87VU6aMYspRYlrUfa3sdHIpN/view?usp=sharing`

It was trained using the Scene Graph Benchmark from `https://github.com/rafiberlin/Scene-Graph-Benchmark.pytorch`,

We used the published pretrained `SGDet, Causal TDE, MOTIFS Model, SUM Fusion` model and followed the instructions
concerning image retrieval under: 
`https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch/blob/master/maskrcnn_benchmark/image_retrieval/S2G-RETRIEVAL.md`
The json files defined in the configuration under the key `scene_graph` were either dowloaded following instructions 
for the Scene Graph Benchmark or were produced training the model (but are not necessary to play the game, only to run 
some evaluation) .

To use one of the avatars for a game session, you need to assigne the desired version in:

`avatar_sgg/scripts/slurk/game_avatar_cli.py`

Either:

`avatar_model = BaselineAvatar(image_directory)`

Or:

`avatar_model = GraphAvatar(image_directory)`


You will also need to download the ADE20K images und some additional annotations for them 
under `https://github.com/clp-research/image-description-sequences`.

Configure the entry `ade20k` in the configuration file accordingly.

# Installation

You can install the scripts to be available from the shell (where the python environment is accessible).

For this simply checkout this repository and perform `python setup.py install` from within the root directory. This will
install the app into the currently activate python environment. After installation, you can use the `game-setup`
, `game-master` and `game-avatar` cli commands (where the python environment is accessible).

**Installation for developers on remote machines**
Run `update.sh` to install the project on a machine. This shell script simply pulls the latest changes and performs the
install from above. As a result, the script will install the python project as an egg
into `$HOME/.local/lib/pythonX.Y/site-packages`.

You have to add the install directory to your python path to make the app
available `export PYTHONPATH=$PYTHONPATH:$HOME/.local/lib/pythonX.Y/site-packages`

Notice: Use the python version X.Y of your choice. Preferebly add this export also to your `.bashrc`.

# Deployment

## A. Run everything on localhost

### Prepare servers and data

#### 1. Start slurk

Checkout `slurk` (revision `6abfd0634f86e21aef10bea84b03ffd0ed7fc6c5`, as the rest API changed afterwards) and run slurk `local_run`. This will start slurk on `localhost:5000`. This will also create the default
admin token.

#### 2. Download and expose the dataset

Download and unpack the ADE20K dataset. Go into the images training directory and start a http server as the server
serving the images. You can use `python -m http.server 8000` for this.

#### 3. Create the slurk game room and player tokens

Checkout `clp-sose21-pm-vision` and run the `game_setup_cli` script or if installed, the `game-setup` cli. By default,
the script expects slurk to run on `localhost:5000`. If slurk runs on another machine, then you must provide
the `--slurk_host` and `--slurk_port` options. If you do not use the default admin token, then you must use
the `--token` option to provide the admin token for the game setup.

The script will create the game room, task and layout using the default admin token via slurks REST API. This will also
create three tokens: one for the game master, player and avatar. See the console output for these tokens. You can also
manually provide a name for the room.

### Prepare clients and bots

#### 1. Start the game master bot

Run the `game_master_cli --token <master-token>` script or the `game-master --token <master-token>` cli. This will
connect the game master with slurk. By default, this will create image-urls that point to `localhost:8000`.

#### 2. Start a browser for the 'Player'

Run a (private-mode) browser and go to `localhost:5000` and login as `Player` using the `<player-token>`.

#### 3. Start the avatar bot

**If the avatar is supposed to be your bot**, then run the `game_avatar_cli --token <avatar-token>` script or the
the `game-avatar --token <avatar-token>` cli. This will start the bot just like with the game master.

Note: This works best, when the game master is joining the room, before the avatar or the player. The order of player
and avatar should not matter. If a game seems not starting or the avatar seems not responding try to restart a game
session with the `/start` command in the chat window.

Another note: The dialog history will be persistent for the room. If you want to "remove" the dialog history, then you
have to create another room using the `game-setup` cli (or restart slurk and redo everything above). The simple avatar
does not keep track of the dialog history.
