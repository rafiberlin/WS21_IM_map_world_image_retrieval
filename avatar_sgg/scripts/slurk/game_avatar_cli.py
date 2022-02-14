"""
    Slurk client for the game master.

    Possibly requires:

    import sys
    sys.path.append("F:\\Development\\git\\clp-sose21-pm-vision")
"""
import base64

import click
import socketIO_client

from avatar_sgg.game_avatar import SimpleAvatar
from avatar_sgg.game_avatar_baseline import BaselineAvatar
from avatar_sgg.game_avatar_graph import GraphAvatar
from avatar_sgg.game_avatar_slurk import AvatarBot


def build_url(host, context=None, port=None, base_url=None, auth=None):
    uri = "http://"
    if auth:
        uri = uri + f"{auth}@"  # seems not to work (try using query parameter)
    uri = uri + f"{host}"
    if context:
        uri = uri + f"/{context}"
    if port:
        uri = uri + f":{port}"
    if base_url:
        uri = uri + f"{base_url}"
    return uri


@click.command()
@click.option("--name", default="None", show_default=True, required=True,
              help="The name suffix for the avatar_sgg. The avatar_sgg will be Avatar-<name> or just Avatar if not given.")
@click.option("--token", show_default=True, required=True,
              help="the token for the avatar_sgg bot. You get this afer game-setup. "
                   "The bot will join the token room.")
@click.option("--slurk_host", default="127.0.0.1", show_default=True, required=True,
              help="domain of the the slurk app")
@click.option("--slurk_context", default=None, show_default=True,
              help="sub-path of to the slurk host")
@click.option("--slurk_port", default="5000", show_default=True, required=True,
              help="port of the slurk app")
@click.option("--image_directory", default="None", show_default=True, required=True,
              help="If images are accessible by the bot, "
                   "then this is the path to the image directory usable as a prefix for images")
def start_and_wait(name, token, slurk_host, slurk_context, slurk_port, image_directory):
    """Start the game master bot."""
    if slurk_port == "None":
        slurk_port = None

    if slurk_port:
        slurk_port = int(slurk_port)

    if image_directory == "None":
        image_directory = None

    if name == "None":
        avatar_name = AvatarBot.NAME
    else:
        avatar_name = AvatarBot.NAME + "-" + name

    custom_headers = {"Authorization": token, "Name": avatar_name}
    socket_url = build_url(slurk_host, slurk_context)
    print("Try to connect to: ", socket_url)
    sio = socketIO_client.SocketIO(socket_url, slurk_port, headers=custom_headers, Namespace=AvatarBot)
    # NOTE: YOU SHOULD REFERENCE YOUR MODEL HERE
    #avatar_model = SimpleAvatar(image_directory)
    #avatar_model = BaselineAvatar(image_directory)
    avatar_model = GraphAvatar(image_directory)
    sio.get_namespace().set_agent(avatar_model)
    print("Connected and everything set. Waiting for incoming traffic...")
    sio.wait()


if __name__ == '__main__':
    start_and_wait()
