import os
import time
import cairosvg


def save(path, content):
    with open(path, "w") as f:
        f.write(content)


def save_svg(cfg, svg_code, svg_id):
    svg_path = f"{cfg.svg_dir}/{svg_id}.svg"
    png_path = f"{cfg.png_dir}/{svg_id}.png"
    svg_code = extract_svg(svg_code)
    save(svg_path, svg_code)
    cairosvg.svg2png(url=svg_path, write_to=png_path, background_color="white")


def read(path):
    with open(path, "r") as f:
        return f.read()


def log(*args, **kwargs):
    time_format = '%Y/%m/%d %H:%M:%S'
    value = time.localtime(int(time.time()))
    formatted = time.strftime(time_format, value)
    print(formatted, *args, **kwargs)


def extract_code(response: str) -> str:
    if "```python" in response:
        response = response.split("```python\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    return response


def extract_svg(response: str) -> str:
    if "```svg\n" in response:
        response = response.split("```svg\n")[1]
        response = response.split("```")[0]
    elif "```svg" in response:
        response = response.split("```svg")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    return response


def extract_json(response: str) -> dict:
    if "```json" in response:
        response = response.split("```json\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    return response


def extract_javascript(response: str) -> dict:
    if "```javascript" in response:
        response = response.split("```javascript\n")[1]
        response = response.split("```")[0]

    response = response.rstrip()

    return response


def get_prompt(target):
    files_to_captions = {
        "astronaut": "An astronaut on the moon",
        "car": "A car",
        "castle": "A castle",
        "cat": "A cat in the sitting position",
        "cottage": "A cozy cottage nestled in a snowy forest",
        "crown": "A crown with jewels",
        "elephant": "An elephant",
        "flamingo": "A walking flamingo",
        "food": "fast food",
        "margarita": "A margarita with a lime wedge",
        "sakura": "A cherry blossom tree",
        "snail": "A snail with a round shell",
        "spaceship": "A spaceship flying in the night sky",
        "tokyo": "A picture of Tokyo",
        "volcano": "A volcano erupting",

        "air_conditioner": "An air conditioner",
        "apple": "A red apple with green leaves and stems",
        "arctic": "An icy arctic landscape",
        "astronaut_horse": "An astronaut riding a horse",
        "avocado_chair": "An avocado shaped chair",
        "bamboo": "Bamboo trees",
        "banana": "A yellow banana",
        "bat": "A black bat flying at night",
        "bell": "A bell with a red ribbon",
        "bubble_tea": "A cup of bubble tea with a straw",
        "bonsai": "A bonsai tree in a pot",
        "bookshelf": "A bookshelf with books",
        "burger": "A cheeseburger with lettuce and tomato",
        "bus": "A yellow school bus",
        "butterfly": "A blue butterfly",
        "cactus_city": "A city with buildings shaped like cacti",
        "cactus": "A tall cactus in a desert",
        "cake": "A chocolate cake with a cherry on top",
        "camera": "A camera",
        "castle": "A castle with tall towers",
        "cat_avatar": "An avatar of a yellow cat",
        "cat_pirate": "A cat dressed as a pirate",
        "cheese": "A piece of cheese with holes",
        "cherry": "Two cherries with stems and leaves",
        "christmas_tree": "A christmas tree with colorful lights",
        "clock_tower": "A clock tower at sunset",
        "cloud": "A single fluffy cloud",
        "coconut": "A coconut with a straw",
        "coffee": "A cup of coffee with steam",
        "compass": "A compass with a gold rim",
        "crab": "A crab extending its claws",
        "daisy": "A daisy flower",
        "desert": "A desert with cacti and rocks",
        "dog_car": "A dog is driving a red pickup truck",
        "dog_chef": "A dog wearing a chef's hat",
        "dog": "A brown dog standing still",
        "donut": "A donut with pink frosting",
        "dragon_bus": "A dragon carrying passengers like a bus",
        "farmhouse": "A farmhouse with a red barn",
        "flask": "A flask with a liquid inside",
        "flower": "A sunflower in bloom",
        "flute": "A silver flute",
        "fox": "A red fox is running",
        "fried_egg": "A fried egg",
        "frog": "A green frog on a lily pad",
        "gate": "A torii gate",
        "gift": "A Christmas gift box",
        "giraffe": "A side view of a giraffe with a long neck",
        "grapes": "A bunch of purple grapes",
        "harbor": "Boats in a harbor",
        "hat": "A magician's hat",
        "headphones": "A pair of headphones",
        "ice_cream": "An ice cream cone with three scoops",
        "island": "A tropical island with palm trees",
        "jungle_temple": "A ruined temple hidden in the jungle",
        "koala": "A koala clinging to a eucalyptus tree",
        "ladybug": "A ladybug on a leaf",
        "lamp": "A table lamp",
        "laptop": "A laptop on a desk",
        "lighthouse": "A lighthouse by the sea",
        "lion": "A lion",
        "lizard": "A lizard on a rock",
        "mailbox": "A red mailbox",
        "maple_leaf": "A red maple leaf",
        "marketplace": "A bustling marketplace",
        "mirror": "A round mirror",
        "mountain_river": "Several mountains behind a river",
        "mountain": "A snowy mountain peak",
        "night_stars": "A night sky full of stars",
        "ocean": "Waves crashing on a beach",
        "octopus": "An octopus swimming underwater with curled tentacles",
        "orange": "Three oranges",
        "owl": "An owl perched on a branch",
        "pancake": "A stack of pancakes with syrup",
        "parrot": "A colorful parrot in flight",
        "peach": "A peach with a leaf",
        "peacock": "A peacock spreading its feathers",
        "pear": "A green pear",
        "penguin": "A penguin standing on ice",
        "piano_forest": "A grand piano in the middle of a forest",
        "piano": "A grand piano",
        "pig": "A pig wearing a backpack",
        "pirate": "A pirate with a parrot on his shoulder",
        "pizza": "A pepperoni pizza slice",
        "plane": "A plane in the sky, with clouds in the background",
        "police_car": "A police car with flashing lights",
        "rabbit": "A gray rabbit with long ears, with carrots near its feet",
        "rainbow": "A rainbow in a clear sky",
        "robot_garden": "A robot tending to a garden",
        "robot": "A cute robot with glowing eyes",
        "roller_coaster": "A roller coaster with loops",
        "sandcastle": "A sandcastle on the beach",
        "shark_balloon": "A balloon shaped like a shark",
        "shield": "A shield with a lion emblem",
        "shovel": "A shovel in a sandbox",
        "signpost": "A signpost with a direction arrow",
        "smartphone": "A smartphone",
        "space_station": "A space station orbiting Earth",
        "spaghetti": "A bowl of spaghetti with meatballs",
        "sphinx": "The Sphinx near the pyramids",
        "spider": "A black spider",
        "squirrel": "Two squirrels standing together",
        "strawberry": "A strawberry",
        "submarine": "A yellow submarine",
        "sushi": "A salmon sushi",
        "sword": "A medieval sword",
        "telescope": "A telescope pointed at the sky",
        "tempura": "A tempura",
        "tent": "A camping tent",
        "train": "A steam train on tracks",
        "trash_bin": "A trash bin with a lid",
        "truck": "A blue delivery truck",
        "violin": "A violin and bow",
        "waterfall": "A waterfall flowing into a river",
        "watermelon": "A slice of watermelon with black seeds",
        "whale": "A whale swimming in the ocean",
        "wolf": "A wolf howling on top of the hill, with a full moon in the sky",
    }
    return files_to_captions[os.path.basename(target)]
