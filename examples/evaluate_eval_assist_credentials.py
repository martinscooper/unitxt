from unitxt import get_logger
from unitxt.api import evaluate, load_dataset
from unitxt.blocks import Task, TaskCard
from unitxt.loaders import LoadFromDictionary
from unitxt.text_utils import print_dict

logger = get_logger()

data = {
    "test": [
        {"context": {"Question": "How is the weather?"}},
    ]
}

criteria = "metrics.llm_as_judge.eval_assist.direct_assessment.criterias.temperature"
metrics = [
    f"metrics.llm_as_judge.eval_assist.direct_assessment.rits.llama3_1_8b[criteria_or_criterias={criteria},option_selection_strategy=PARSE_OUTPUT_TEXT,"
    "inference_engine=engines.rits.llama_3_1_70b_instruct[model_name=openai/meta-llama/llama-3-1-70b-instruct,credentials={api_key=}]]"
]
card = TaskCard(
    loader=LoadFromDictionary(data=data, data_classification_policy=["public"]),
    task=Task(
        input_fields={"context": dict},
        reference_fields={},
        prediction_type=str,
        metrics=metrics,
    ),
)

test_dataset = load_dataset(card=card, template="templates.empty")["test"]

predictions = [
    """On most days, the weather is warm and humid, with temperatures often soaring into the high 80s and low 90s Fahrenheit (around 31-34°C). The dense foliage of the jungle acts as a natural air conditioner, keeping the temperature relatively stable and comfortable for the inhabitants.""",
]

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