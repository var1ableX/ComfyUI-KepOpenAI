from typing import Tuple

import torch
from openai import Client as OpenAIClient

from .lib import credentials, image

openAI_models = None
#region chatGPTDefaultInitMessages
chatGPTDefaultInitMessage_tags = """
First, some basic Stable Diffusion prompting rules for you to better understand the syntax. The parentheses are there for grouping prompt words together, so that we can set uniform weight to multiple words at the same time. Notice the ":1.2" in (masterpiece, best quality, absurdres:1.2), it means that we set the weight of both "masterpiece" and "best quality" to 1.2. The parentheses can also be used to directly increase weight for single word without adding ":WEIGHT". For example, we can type ((masterpiece)), this will increase the weight of "masterpiece" to 1.21. This basic rule is imperative that any parentheses in a set of prompts have purpose, and so they must not be remove at any case. Conversely, when brackets are used in prompts, it means to decrease the weight of a word. For example, by typing "[bird]", we decrease the weight of the word "bird" by 1.1.
Now, I've develop a prompt template to use generate character portraits in Stable Diffusion. Here's how it works. Every time user sent you "CHAR prompts", you should give prompts that follow below format:
CHAR: [pre-defined prompts], [location], [time], [weather], [gender], [skin color], [photo type], [pose], [camera position], [facial expression], [body feature], [skin feature], [eye color], [outfit], [hair style], [hair color], [accessories], [random prompt],

[pre-defined prompts] are always the same, which are "RAW, (masterpiece, best quality, photorealistic, absurdres, 8k:1.2), best lighting, complex pupils, complex textile, detailed background". Don't change anything in [pre-defined prompts], meaning that you SHOULD NOT REMOVE OR MODIFY the parentheses since their purpose is for grouping prompt words together so that we can set uniform weight to them;
[location] is the location where character is in, can be either outdoor location or indoor, but need to be specific;
[time] refers to the time of day, can be "day", "noon", "night", "evening", "dawn" or "dusk";
[weather] is the weather, for example "windy", "rainy" or "cloudy";
[gender] is either "1boy" or "1girl";
[skin color] is the skin color of the character, could be "dark skin", "yellow skin" or "pale skin";
[photo type] can be "upper body", "full body", "close up", "mid-range", "Headshot", "3/4 shot" or "environmental portrait";
[pose] is the character's pose, for example, "standing", "sitting", "kneeling" or "squatting" ...;
[camera position] can be "from top", "from below", "from side", "from front" or "from behind";
[facial expression] is the expression of the character, you should give user a random expression;
[body feature] describe how the character's body looks like, for example, it could be "wide hip", "large breasts" or "sexy", try to be creative;
[skin feature] is the feature of character's skin. Could be "scar on skin", "dirty skin", "tanned mark", "birthmarks" or other skin features you can think of;
[eye color] is the pupil color of the character, it can be of any color as long as the color looks natural on human eyes, so avoid colors like pure red or pure black;
[outfit] is what character wears, it should include at least the top wear, bottom wear and footwear, for example, "crop top, shorts, sneakers", the style of outfit can be any, but the [character gender] should be considered;
[hair style] is the hairstyle of the character, [character gender] should be taken into account when setting the hairstyle;
[hair color] can be of any color, for example, "orange hair", "multi-colored hair";
[accessories] is the accessory the character might wear, can be "chocker", "earrings", "bracelet" or other types of accessory;
[random prompt] will test your creativity, put anything here, just remember that you can only use nouns in [random prompt], the number of [random prompt] can be between 1 to 4. For example, you could give "campfire", but you can also give "shooting star, large moon, fallen leaves". Again, be creative with this one.

also use gelbooru  tags as much as you can
if you use gelbooru  write "gTags" before it 
Do not use markdown syntax in prompts, do not use capital letter and keep all prompt words in the same line. Respond with "prompt:" to start prompting with us.

""";

chatGPTDefaultInitMessage_description = """
We are going to create a function. USER will provide you with a simple statement including a Format and you the "SYSTEM" will respond with a highly emotive prompt suitable for image generation. Response format: a [format] of a [insert emotive prompt], IN THE STYLE OF [Artist] Default Format = PHOTO. If Artist is provided then postfix appropriately based otherwise skip. 

Examples:
 
USER= What is the scariest thing you found at the local flea market? Format: ILLUSTRATION, Artist: Andrew Catherall
SYSTEM: an ILLUSTRATION of an old, weathered doll with piercing glass eyes, dressed in Victorian-era clothing, found sitting alone on a dusty shelf at the local flea market. IN THE STYLE OF Andrew Catherall

USER= A cityscape during the golden hour, skyscrapers basking in the sun's glow. Format: PHOTO (City:1.2), (Golden Hour:1.1), Skyscrapers, Artist: Monet
SYSTEM: A PHOTO of a city basking in the golden hour, the skyscrapers aglow with the sun's warmth, emanating a sense of peace and tranquility in photorealistic, high-resolution detail. (City:1.2), (Golden Hour:1.1), Skyscrapers, IN THE STYLE OF Monet

USER= A cityscape during the golden hour, skyscrapers basking in the sun's glow. Format: SKETCH, Artist: Van Gogh
SYSTEM: A SKETCH of a city basking in the golden hour, the skyscrapers aglow with the sun's warmth, emanating a sense of peace and tranquility in photorealistic, high-resolution detail. IN THE STYLE OF Van Gogh

USER= A forest in spring, sunlight dappling the vibrant greenery. (Forest:1.3), Sunlight, ((Greenery)). Format: CARTOON, Artist: Chuck Schmidt
SYSTEM: A CARTOON of a forest in spring, sunlight dappling the verdant foliage, the interplay of light and shadow evoking a sense of rejuvenation in photorealistic, high-resolution detail. (Forest:1.3), Sunlight, ((Greenery)), IN THE STYLE OF Chuck Schmidt

USER= A forest in spring, sunlight dappling the vibrant greenery.
SYSTEM: A PHOTO of a forest in spring, sunlight dappling the verdant foliage, the interplay of light and shadow evoking a sense of rejuvenation in photorealistic, high-resolution detail.

Using terse language and no less than {num_terms} terms, generate an image prompt from:

USER=  
""";

class CC:
    CLEAN = '\33[0m'
    BOLD = '\33[1m'
    ITALIC = '\33[3m'
    UNDERLINE = '\33[4m'
    BLINK = '\33[5m'
    BLINK2 = '\33[6m'
    SELECTED = '\33[7m'

    BLACK = '\33[30m'
    RED = '\33[31m'
    GREEN = '\33[32m'
    YELLOW = '\33[33m'
    BLUE = '\33[34m'
    VIOLET = '\33[35m'
    BEIGE = '\33[36m'
    WHITE = '\33[37m'

    GREY = '\33[90m'
    LIGHTRED = '\33[91m'
    LIGHTGREEN = '\33[92m'
    LIGHTYELLOW = '\33[93m'
    LIGHTBLUE = '\33[94m'
    LIGHTVIOLET = '\33[95m'
    LIGHTBEIGE = '\33[96m'
    LIGHTWHITE = '\33[97m'

def get_init_message(isTags=False):
    if(isTags):
        return chatGPTDefaultInitMessage_tags
    else:
        return chatGPTDefaultInitMessage_description

#endregion chatGPTDefaultInitMessages


class O_ChatGPT_O:
    """
    this node is based on the openAI GPT-3 API to generate propmpts using the AI
    """
    # Define the input types for the node
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Multiline string input for the prompt
                "prompt": ("STRING", {"multiline": True}),
                "model": (["gpt-4"], {"default": "gpt-4"}),
                "max_words": ("INT", {"min": 1, "max": 150, "default": 40}),
                "behaviour": (["tags","description"], {"default": "description"}),
                "print_to_console": ([False, True],),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING",)  # Define the return type of the node
    FUNCTION = "fun"  # Define the function name for the node
    CATEGORY = "OpenAI"  # Define the category for the node

    def __init__(self):
        self.open_ai_client: OpenAIClient = OpenAIClient(
            api_key=credentials.get_open_ai_api_key()
        )

    def fun(self, model, prompt,behaviour, seed, max_words, print_to_console):
        # install_openai()  # Install the OpenAI module if not already installed
        # from openai import OpenAI

          # Set the API key for the OpenAI module
        initMessage = "";
        if(behaviour == "description"):
            initMessage = get_init_message(False);
            #initMessage.format(num_terms=max_words)
            initMessage = initMessage.replace("num_terms", str(max_words))
            if(print_to_console):
                print(f'{CC.VIOLET}{initMessage}')
        else:
            initMessage = get_init_message(True);
        # Create a chat completion using the OpenAI module
        try:
            completion = self.open_ai_client.chat.completions.create(model=model,
            messages=[
                {"role": "user", "content":initMessage},
                {"role": "user", "content": prompt}
            ])
        except:  # sometimes it fails first time to connect to server
            completion = self.open_ai_client.chat.completions.create(model=model,
            messages=[
                {"role": "user", "content": initMessage},
                {"role": "user", "content": prompt}
            ])
        # Get the answer from the chat completion
        if len(completion.choices) == 0:
            raise Exception("No response from OpenAI API")
        answer = completion.choices[0].message.content
        #print(f"return type is: {type(answer)}")
        return (answer,)
    

class concat_text_O:
    """
    This node will concatenate two strings together
    """
    @ classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text1": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "text2": ("STRING", {"multiline": True, "defaultBehavior": "input"}),
            "separator": ("STRING", {"multiline": False, "default": ","}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "fun"
    CATEGORY = "OpenAI"

    @ staticmethod
    def fun(text1, separator, text2):
        return (text1 + separator + text2,)

class ImageWithPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Image": ("IMAGE", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "Generate a high quality caption for the image. The most important aspects of the image should be described first. If needed, weights can be applied to the caption in the following format: '(word or phrase:weight)', where the weight should be a float less than 2.",
                    },
                ),
                "max_tokens": ("INT", {"min": 1, "max": 2048, "default": 77}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_completion"
    CATEGORY = "OpenAI"

    def __init__(self):
        self.open_ai_client: OpenAIClient = OpenAIClient(
            api_key=credentials.get_open_ai_api_key()
        )

    def generate_completion(
        self, Image: torch.Tensor, prompt: str, max_tokens: int
    ) -> Tuple[str]:
        b64image = image.pil2base64(image.tensor2pil(Image))
        response = self.open_ai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64image}"},
                        },
                    ],
                }
            ],
        )
        if len(response.choices) == 0:
            raise Exception("No response from OpenAI API")

        return (response.choices[0].message.content,)
