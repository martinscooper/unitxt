from src.unitxt.catalog import add_to_catalog
from src.unitxt.formats import SystemFormat

format = SystemFormat(
    model_input_format="<|system|>\n{system_prompt}\n<|user|>\n{instruction}\n{source}\n<|assistant|>\n{target_prefix}",
)

add_to_catalog(format, "formats.models.labrador.zero_shot", overwrite=True)

format = SystemFormat(
    demo_format="{source}\n{target_prefix}\n{target}\n\n",
    model_input_format=(
        "<|system|>\n"
        "{system_prompt}\n"
        "<|user|>\n"
        "{instruction}\n"
        "Your response should only include the answer. Do not provide any further explanation.\n"
        "\n"
        "Here are some examples, complete the last one:\n"
        "{demos}"
        "{source}\n"
        "{target_prefix}\n"
        "<|assistant|>\n"
    ),
)

add_to_catalog(format, "formats.models.labrador.few_shot", overwrite=True)

