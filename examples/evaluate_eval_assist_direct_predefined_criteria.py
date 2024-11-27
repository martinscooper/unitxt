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
    f"metrics.llm_as_judge.eval_assist.direct_assessment.rits.llama3_1_70b[criteria_or_criterias={criteria}]"
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
    """you are""",
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
