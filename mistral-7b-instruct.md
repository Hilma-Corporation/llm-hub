# Mistral-7B-Instruct-v0.3

Voici la version mise à jour avec des icônes emoji pour les fichiers safe tensor :

* [.gitattributes](https://gofile.me/7srKA/LuwELsZ4z) - 1.52 kB

* [README.md](https://gofile.me/7srKA/CtnUGOnql) - 5.86 kB

* [config.json](https://gofile.me/7srKA/XpRgI2BFD) - 601 Bytes

* [consolidated.safetensors](https://gofile.me/7srKA/Pb4O58jZW) 🛡️ - 14.5 GB

* [generation_config.json](https://gofile.me/7srKA/FQwiBqHfN) - 116 Bytes

* [model-00001-of-00003.safetensors](https://gofile.me/7srKA/Pz55FIN0p) 🛡️ - 4.95 GB

* [model-00002-of-00003.safetensors](https://gofile.me/7srKA/JMT1n83D6) 🛡️ - 5 GB

* [model-00003-of-00003.safetensors](https://gofile.me/7srKA/jSXiO3IJN) 🛡️ - 4.55 GB

* [model.safetensors.index.json](https://gofile.me/7srKA/OQEUaKhyx) - 24 kB

* [params.json](https://gofile.me/7srKA/IwatAhWFv) - 202 Bytes

* [special_tokens_map.json](https://gofile.me/7srKA/pIliKPtee) - 414 Bytes

* [tokenizer.json](https://gofile.me/7srKA/HVTxWfBRS) - 1.96 MB

* [tokenizer.model](https://gofile.me/7srKA/0JAaVgbBR) - 587 kB

* [tokenizer.model.v3](https://gofile.me/7srKA/taKG9WEII) - 587 kB

* [tokenizer_config.json](https://gofile.me/7srKA/SDpWj1K8I) - 137 kB


Si vous avez besoin d'autres modifications ou ajouts, n'hésitez pas à le demander.



The Mistral-7B-Instruct-v0.3 Large Language Model (LLM) is an instruct fine-tuned version of the Mistral-7B-v0.3.

Mistral-7B-v0.3 has the following changes compared to [Mistral-7B-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/edit/main/README.md)
- Extended vocabulary to 32768
- Supports v3 Tokenizer
- Supports function calling

## Installation

It is recommended to use `mistralai/Mistral-7B-Instruct-v0.3` with [mistral-inference](https://github.com/mistralai/mistral-inference). For HF transformers code snippets, please keep scrolling.

```
pip install mistral_inference
```

## Download

```py
from huggingface_hub import snapshot_download
from pathlib import Path

mistral_models_path = Path.home().joinpath('mistral_models', '7B-Instruct-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)
```

### Chat

After installing `mistral_inference`, a `mistral-chat` CLI command should be available in your environment. You can chat with the model using

```
mistral-chat $HOME/mistral_models/7B-Instruct-v0.3 --instruct --max_tokens 256
```

### Instruct following

```py
from mistral_inference.model import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

completion_request = ChatCompletionRequest(messages=[UserMessage(content="Explain Machine Learning to me in a nutshell.")])

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
```

### Function calling

```py
from mistral_common.protocol.instruct.tool_calls import Function, Tool
from mistral_inference.model import Transformer
from mistral_inference.generate import generate

from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest


tokenizer = MistralTokenizer.from_file(f"{mistral_models_path}/tokenizer.model.v3")
model = Transformer.from_folder(mistral_models_path)

completion_request = ChatCompletionRequest(
    tools=[
        Tool(
            function=Function(
                name="get_current_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "The temperature unit to use. Infer this from the users location.",
                        },
                    },
                    "required": ["location", "format"],
                },
            )
        )
    ],
    messages=[
        UserMessage(content="What's the weather like today in Paris?"),
        ],
)

tokens = tokenizer.encode_chat_completion(completion_request).tokens

out_tokens, _ = generate([tokens], model, max_tokens=64, temperature=0.0, eos_id=tokenizer.instruct_tokenizer.tokenizer.eos_id)
result = tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

print(result)
```

## Generate with `transformers`

If you want to use Hugging Face `transformers` to generate text, you can do something like this.

```py
from transformers import pipeline

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]
chatbot = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.3")
chatbot(messages)
```

## Limitations

The Mistral 7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance. 
It does not have any moderation mechanisms. We're looking forward to engaging with the community on ways to
make the model finely respect guardrails, allowing for deployment in environments requiring moderated outputs.

## The Mistral AI Team

Albert Jiang, Alexandre Sablayrolles, Alexis Tacnet, Antoine Roux, Arthur Mensch, Audrey Herblin-Stoop, Baptiste Bout, Baudouin de Monicault, Blanche Savary, Bam4d, Caroline Feldman, Devendra Singh Chaplot, Diego de las Casas, Eleonore Arcelin, Emma Bou Hanna, Etienne Metzger, Gianna Lengyel, Guillaume Bour, Guillaume Lample, Harizo Rajaona, Jean-Malo Delignon, Jia Li, Justus Murke, Louis Martin, Louis Ternon, Lucile Saulnier, Lélio Renard Lavaud, Margaret Jennings, Marie Pellat, Marie Torelli, Marie-Anne Lachaux, Nicolas Schuhl, Patrick von Platen, Pierre Stock, Sandeep Subramanian, Sophia Yang, Szymon Antoniak, Teven Le Scao, Thibaut Lavril, Timothée Lacroix, Théophile Gervet, Thomas Wang, Valera Nemychnikova, William El Sayed, William Marshall

[tokenizer.model.v3](https://gofile.me/7srKA/taKG9WEII)
