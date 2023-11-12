from .nodes import ImageWithPrompt
from .nodes import O_ChatGPT_O
from .nodes import concat_text_O

NODE_CLASS_MAPPINGS = {
    "KepOpenAI_ImageWithPrompt": ImageWithPrompt,
    "KepOpenAI_O_ChatGPT_O": O_ChatGPT_O,
    "KepOpenAI_concat_text_O": concat_text_O,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "KepOpenAI_ImageWithPrompt": "Image With Prompt",
    "KepOpenAI_O_ChatGPT_O": "ChatGPT Prompt Simple",
    "KepOpenAI_concat_text_O": "ChatGPT Concat Helper"
}
