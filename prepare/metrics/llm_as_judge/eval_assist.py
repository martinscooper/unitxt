from templates_eval_assist_direct import direct_assessment_template_dict
from templates_eval_assist_pairwise import pairwise_comparison_template_dict
from unitxt import add_to_catalog, get_logger
from unitxt.eval_assist_constants import (
    DIRECT_ASSESSMENT_CRITERIAS,
    EVALUATOR_TO_MODEL_ID,
    EVALUATORS_METADATA,
    INFERENCE_ENGINE_NAME_TO_CLASS,
    PAIRWISE_COMPARISON_CRITERIAS,
    PROVIDER_TO_STRATEGY,
    EvaluatorNameEnum,
    EvaluatorTypeEnum,
    ModelFamilyEnum,
    ModelProviderEnum,
    OptionSelectionStrategyEnum,
)
from unitxt.eval_assist_llm_as_judge_direct import EvalAssistLLMAsJudgeDirect
from unitxt.eval_assist_llm_as_judge_pairwise import EvalAssistLLMAsJudgePairwise
from unitxt.eval_assist_utils import get_evaluator_metadata
from unitxt.inference import MockInferenceEngine

logger = get_logger()


def get_evaluator(
    name: EvaluatorNameEnum,
    evaluator_type: EvaluatorTypeEnum,
    provider: ModelProviderEnum,
    mock_inference_engine: bool = False
) -> EvalAssistLLMAsJudgeDirect | EvalAssistLLMAsJudgePairwise:
    evaluator_metadata = get_evaluator_metadata(name)

    inference_params = (
        {"max_tokens": 1024, "seed": 42}
        if provider in [ModelProviderEnum.RITS, ModelProviderEnum.OPENAI]
        else {"max_new_tokens": 1024, "random_seed": 42}
    )

    params = {
        "model_name": EVALUATOR_TO_MODEL_ID[evaluator_metadata.name],
        **inference_params,
    }

    inference_engine = INFERENCE_ENGINE_NAME_TO_CLASS[provider](**params) 

    params = {
        "inference_engine": inference_engine,
        "option_selection_strategy": PROVIDER_TO_STRATEGY[provider].name,
        "evaluator_name": evaluator_metadata.name.name,
        "model_family": evaluator_metadata.model_family.name,
    }

    evaluator_klass = (
        EvalAssistLLMAsJudgeDirect
        if evaluator_type == EvaluatorTypeEnum.DIRECT_ASSESSMENT
        else EvalAssistLLMAsJudgePairwise
    )

    return evaluator_klass(**params)


logger.debug("Registering criterias...")
# Register all the predefined criterisa
for criteria in DIRECT_ASSESSMENT_CRITERIAS:
    add_to_catalog(
        criteria,
        f"metrics.llm_as_judge.eval_assist.direct_assessment.criterias.{criteria.name}",
        overwrite=True,
    )

for criteria in PAIRWISE_COMPARISON_CRITERIAS:
    add_to_catalog(
        criteria,
        f"metrics.llm_as_judge.eval_assist.pairwise_comparison.criterias.{criteria.name}",
        overwrite=True,
    )

logger.debug("Registering evaluators...")
for evaluator_metadata in EVALUATORS_METADATA:
    for provider in evaluator_metadata.providers:
        for evaluator_type in [
            EvaluatorTypeEnum.DIRECT_ASSESSMENT,
            EvaluatorTypeEnum.PAIRWISE_COMPARISON,
        ]:
            evaluator = get_evaluator(
                name=evaluator_metadata.name,
                evaluator_type=evaluator_type,
                provider=provider,
            )

            metric_name = (
                evaluator_metadata.name.value.lower()
                .replace("-", "_")
                .replace(".", "_")
                .replace(" ", "_")
            )
            add_to_catalog(
                evaluator,
                f"metrics.llm_as_judge.eval_assist.{evaluator_type.value}.{provider.value.lower()}.{metric_name}",
                overwrite=True,
            )

for evaluator_type in [
            EvaluatorTypeEnum.DIRECT_ASSESSMENT,
            EvaluatorTypeEnum.PAIRWISE_COMPARISON,
        ]:
    
    evaluator_klass = (
        EvalAssistLLMAsJudgeDirect
        if evaluator_type == EvaluatorTypeEnum.DIRECT_ASSESSMENT
        else EvalAssistLLMAsJudgePairwise
    )

    e = evaluator_klass(
        inference_engine=MockInferenceEngine(model_name='mock'),
        option_selection_strategy=OptionSelectionStrategyEnum.PARSE_OUTPUT_TEXT.name,
        evaluator_name="",
        model_family="",
    )

    add_to_catalog(
        e,
        f"metrics.llm_as_judge.eval_assist.{evaluator_type.value}",
        overwrite=True,
    )