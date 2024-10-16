from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.templates import InputOutputTemplate, TemplatesDict
from unitxt.text_utils import print_dict
logger = get_logger()
data = {
    "test": [
       {"context": "How is the weather?"},
        {"context": "How is the weather?"},
        {"context": "How is the weather?"}
    ]
}

pairwise_criteria = "metrics.llm_as_judge.eval_assist.pairwise.criterias.temperature"
metric = f"metrics.llm_as_judge.eval_assist.pairwise.mixtral_8_7b[pairwise_criteria={pairwise_criteria}]"

card = TaskCard(
    loader=LoadFromDictionary(data=data),
    task=Task(
        input_fields={"context": str},
        reference_fields={},
        prediction_type=str,
        metrics = [metric]
    ),
    templates=TemplatesDict(
        {
            "simple": InputOutputTemplate(
                input_format="{context}",
                output_format="",
            )
        }
    ),
)

test_dataset = load_dataset(card=card, template_card_index="simple")["test"]

# list[list(str)] : each pair contains response from both the models
predictions = [["""On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants."""],
    ["""On most days, the weather is warm and humid. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit. The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants."""]]
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

for instance in evaluated_dataset:
    print_dict(
        instance,
        keys_to_print=[
            "source",
            "prediction",
            "processed_prediction",
            "references",
            "score",
        ],
    )
