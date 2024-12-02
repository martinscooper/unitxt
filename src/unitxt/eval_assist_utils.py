from typing import TypedDict, Union
from unitxt.eval_assist_constants import (
    EVALUATORS_METADATA,
    DirectCriteriaCatalogEnum,
    EvaluatorMetadata,
    EvaluatorNameEnum,
    EvaluatorTypeEnum,
    ModelProviderEnum,
    PairwiseComparisonCriteriaCatalogEnum,
    MODEL_RENAMINGS,
)

def get_parsed_context(context: dict[str, str]):
    return (
        "\n".join([f"{key}: {value}" for key, value in context.items()])
        if len(context) > 1
        or len(context) == 0
        or list(context.keys())[0].lower() != "context"
        else context[list(context.keys())[0]]
    )


def get_evaluator_metadata(
    name: EvaluatorNameEnum
) -> EvaluatorMetadata:  # , evaluator_type: EvaluatorTypeEnum) -> EvaluatorMetadata:
    evaluator_search = [
        e for e in EVALUATORS_METADATA if e.name == name
    ]  # and e.evaluator_type == evaluator_type]
    if len(evaluator_search) == 0:
        # raise ValueError(f'A {evaluator_type} evaluator with id {name} does not exist.')
        raise ValueError(f"An evaluator with id {name} does not exist.")
    if len(evaluator_search) > 1:
        # raise ValueError(f'A {evaluator_type} evaluator with id {name} matched several models.')
        raise ValueError(f"An evaluator with id {name} matched several models.")
    return evaluator_search[0]


def get_criteria(
    type: EvaluatorTypeEnum,
    name: DirectCriteriaCatalogEnum | PairwiseComparisonCriteriaCatalogEnum,
):
    try:
        return (
            DirectCriteriaCatalogEnum
            if type == EvaluatorTypeEnum.DIRECT_ASSESSMENT
            else PairwiseComparisonCriteriaCatalogEnum
        )[name]
    except:
        raise Exception("The provided criteria type and name does not exist")


class CatalogDefinition(TypedDict):
    name: str
    params: dict[str, Union[str, dict[str, str], "CatalogDefinition"]]

def parse_catalog_definition(catalog_definition: CatalogDefinition):
     parsed_parts = []
     result = catalog_definition['name']
     for key, value in catalog_definition['params'].items():
          if isinstance(value, dict) and 'params' in value:
              parsed_parts.append(f"{key}={parse_catalog_definition(value)}")
          elif isinstance(value, dict):
               nested_parts = ",".join(f"{k}={v}" for k, v in value.items())
               parsed_parts.append(f"{key}={{{nested_parts}}}")
          else:
               parsed_parts.append(f"{key}={value}")
               
     result = f"{result}[{','.join(parsed_parts)}]"

     return result

def rename_model_if_required(model_name: str, provider: ModelProviderEnum) -> str:
    if provider in MODEL_RENAMINGS and model_name in MODEL_RENAMINGS[provider]:
            return MODEL_RENAMINGS[provider][model_name]
    return model_name

if __name__ == "__main__":
    # example of using a catalog definition object
    catalog_definition: CatalogDefinition = {
        'name': 'metrics.llm_as_judge.eval_assist.direct_assessment.rits.llama3_1_8b',
        'params': {
            'criteria_or_criterias': "metrics.llm_as_judge.eval_assist.direct_assessment.criterias.temperature",
            'inference_engine': {
                'name': 'engines.ibm_wml.llama_3_70b_instruct',
                'params': {
                    'model_name': 'meta-llama/llama-3-1-70b-instruct',
                    'credentials': {
                        'url': '',
                        'project_id': '',
                        'apikey': ''
                    }
                }
            }
        }
    }
    parsed = parse_catalog_definition(catalog_definition)
    print(parsed)