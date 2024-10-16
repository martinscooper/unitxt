import abc
import dataclasses
import os
import re
from typing import Any, Dict, List, Literal, Optional, Union

from datasets import DatasetDict
from tqdm import tqdm

from .artifact import Artifact, fetch_artifact
from .dataclass import InternalField, NonPositionalField
from .deprecation_utils import deprecation
from .image_operators import extract_images
from .logging_utils import get_logger
from .operator import PackageRequirementsMixin
from .settings_utils import get_settings
from .type_utils import isoftype

settings = get_settings()


def get_model_and_label_id(model_name, label):
    model_id = model_name.split("/")[-1].replace("-", "_").replace(".", ",").lower()
    return f"{model_id}_{label}"


@dataclasses.dataclass
class TextGenerationInferenceOutput:
    """Contains the prediction results and metadata for the inference.

    Args:
    prediction (Union[str, List[Dict[str, Any]]]): If this is the result of an _infer call, the string predicted by the model.
    If this is the results of an _infer_log_probs call, a list of dictionaries. The i'th dictionary represents
    the i'th token in the response. The entry "top_tokens" in the dictionary holds a sorted list of the top tokens
    for this position and their probabilities.
    For example: [ {.. "top_tokens": [ {"text": "a", 'logprob': },  {"text": "b", 'logprob': } ....]},
                   {.. "top_tokens": [ {"text": "c", 'logprob': },  {"text": "d", 'logprob': } ....]}
                ]

    input_tokens (int) : number of input tokens to the model.
    output_tokens (int) : number of output tokens to the model.
    stop_reason (str): stop reason for text generation, for example "eos" (end of string).
    seed (int): seed used by the model during generation.
    input_text (str): input to the model.
    model_name (str): the model_name as kept in the InferenceEngine.
    inference_type (str): The label stating the type of the InferenceEngine.
    """

    prediction: Union[str, List[Dict[str, Any]]]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    seed: Optional[int] = None
    input_text: Optional[str] = None
    model_name: Optional[str] = None
    inference_type: Optional[str] = None


class InferenceEngine(abc.ABC, Artifact):
    """Abstract base class for inference."""

    @abc.abstractmethod
    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        """Perform inference on the input dataset.

        If return_meta_data - returns a list of TextGenerationInferenceOutput, else returns a list of the string.
        return_meta_data is only supported for some InferenceEngines.
        predictions.
        """
        pass

    @abc.abstractmethod
    def prepare_engine(self):
        """Perform inference on the input dataset."""
        pass

    def prepare(self):
        if not settings.mock_inference_mode:
            self.prepare_engine()

    def infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        """Verifies instances of a dataset and perform inference on the input dataset.

        If return_meta_data - returns a list of TextGenerationInferenceOutput, else returns a list of the string
        predictions.
        """
        if return_meta_data and not hasattr(self, "get_return_object"):
            raise NotImplementedError(
                f"Inference engine {self.__class__.__name__} does not support return_meta_data as it "
                f"does not contain a 'get_return_object' method. Please set return_meta_data=False."
            )

        [self.verify_instance(instance) for instance in dataset]
        if settings.mock_inference_mode:
            return [instance["source"] for instance in dataset]
        return self._infer(dataset, return_meta_data)

    def get_engine_id(self):
        raise NotImplementedError()

    @deprecation(version="2.0.0")
    def _set_inference_parameters(self):
        """Sets inference parameters of an instance based on 'parameters' attribute (if given)."""
        if hasattr(self, "parameters") and self.parameters is not None:
            get_logger().warning(
                f"The 'parameters' attribute of '{self.get_pretty_print_name()}' "
                f"is deprecated. Please pass inference parameters directly to the "
                f"inference engine instance instead."
            )

            for param, param_dict_val in self.parameters.to_dict(
                [self.parameters]
            ).items():
                param_inst_val = getattr(self, param)
                if param_inst_val is None:
                    setattr(self, param, param_dict_val)

    def get_model_details(self) -> Dict:
        """Might not be possible to implement for all inference engines. Returns an empty dict by default."""
        return {}


class LogProbInferenceEngine(abc.ABC, Artifact):
    """Abstract base class for inference with log probs."""

    @abc.abstractmethod
    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        """Perform inference on the input dataset  that returns log probs.

        If return_meta_data - returns a list of TextGenerationInferenceOutput, else returns a list of the logprob dicts.
        return_meta_data is only supported for some InferenceEngines.
        predictions.
        """
        pass

    def infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        """Verifies instances of a dataset and performs inference that returns log probabilities of top tokens.

        For each instance , generates a list of top tokens per position.
        [ "top_tokens": [ { "text": ..., "logprob": ...} , ... ]
        If return_meta_data - returns a list of TextGenerationInferenceOutput, else returns the list of the logprob dicts.
        return_meta_data is only supported for some InferenceEngines.
        """
        if return_meta_data and not hasattr(self, "get_return_object"):
            raise NotImplementedError(
                f"Inference engine {self.__class__.__name__} does not support return_meta_data as it "
                f"does not contain a 'get_return_object' method. Please set return_meta_data=False."
            )

        [self.verify_instance(instance) for instance in dataset]
        return self._infer_log_probs(dataset, return_meta_data)


class LazyLoadMixin(Artifact):
    lazy_load: bool = NonPositionalField(default=False)

    @abc.abstractmethod
    def _is_loaded(self):
        pass


class HFPipelineBasedInferenceEngine(
    InferenceEngine, PackageRequirementsMixin, LazyLoadMixin
):
    model_name: str
    max_new_tokens: int
    use_fp16: bool = True

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers"
    }

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, "hf_pipeline")

    def _prepare_pipeline(self):
        import torch
        from transformers import AutoConfig, pipeline

        model_args: Dict[str, Any] = (
            {"torch_dtype": torch.float16} if self.use_fp16 else {}
        )
        model_args.update({"max_new_tokens": self.max_new_tokens})

        device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else 0
            if torch.cuda.is_available()
            else "cpu"
        )
        # We do this, because in some cases, using device:auto will offload some weights to the cpu
        # (even though the model might *just* fit to a single gpu), even if there is a gpu available, and this will
        # cause an error because the data is always on the gpu
        if torch.cuda.device_count() > 1:
            assert device == torch.device(0)
            model_args.update({"device_map": "auto"})
        else:
            model_args.update({"device": device})

        task = (
            "text2text-generation"
            if AutoConfig.from_pretrained(
                self.model_name, trust_remote_code=True
            ).is_encoder_decoder
            else "text-generation"
        )

        if task == "text-generation":
            model_args.update({"return_full_text": False})

        self.model = pipeline(
            model=self.model_name, trust_remote_code=True, **model_args
        )

    def prepare_engine(self):
        if not self.lazy_load:
            self._prepare_pipeline()

    def _is_loaded(self):
        return hasattr(self, "model") and self.model is not None

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if not self._is_loaded():
            self._prepare_pipeline()

        outputs = []
        for output in self.model([instance["source"] for instance in dataset]):
            if isinstance(output, list):
                output = output[0]
            outputs.append(output["generated_text"])
        return outputs


def mock_logprobs_default_value_factory() -> List[Dict[str, Any]]:
    return [
        {
            "logprob": -1,
            "text": "[[10]]",
            "top_tokens": [
                {"logprob": -1, "text": "[[10]]"},
            ],
        }
    ]


class MockInferenceEngine(InferenceEngine, LogProbInferenceEngine):
    model_name: str
    default_inference_value: str = "[[10]]"
    default_inference_value_logprob: List[Dict[str, Any]] = dataclasses.field(
        default_factory=mock_logprobs_default_value_factory,
    )
    label: str = "mock_inference_engine"

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, "mock")

    def prepare_engine(self):
        return

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return [
            self.get_return_object(
                self.default_inference_value, instance, return_meta_data
            )
            for instance in dataset
        ]

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        return [
            self.get_return_object(
                self.default_inference_value_logprob, instance, return_meta_data
            )
            for instance in dataset
        ]

    def get_return_object(self, predict_result, instance, return_meta_data):
        if return_meta_data:
            return TextGenerationInferenceOutput(
                prediction=predict_result,
                input_tokens=len(instance["source"]),
                output_tokens=len(predict_result),
                model_name=self.model_name,
                inference_type=self.label,
                input_text=instance["source"],
                seed=111,
                stop_reason="",
            )
        return predict_result


class MockModeMixin(Artifact):
    mock_mode: bool = False


class IbmGenAiInferenceEngineParamsMixin(Artifact):
    beam_width: Optional[int] = None
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    include_stop_sequence: Optional[bool] = None
    length_penalty: Any = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_options: Any = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    time_limit: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    truncate_input_tokens: Optional[int] = None
    typical_p: Optional[float] = None


@deprecation(version="2.0.0", alternative=IbmGenAiInferenceEngineParamsMixin)
class IbmGenAiInferenceEngineParams(Artifact):
    beam_width: Optional[int] = None
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    include_stop_sequence: Optional[bool] = None
    length_penalty: Any = None
    max_new_tokens: Optional[int] = None
    min_new_tokens: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    return_options: Any = None
    stop_sequences: Optional[List[str]] = None
    temperature: Optional[float] = None
    time_limit: Optional[int] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    truncate_input_tokens: Optional[int] = None
    typical_p: Optional[float] = None


class GenericInferenceEngine(InferenceEngine):
    default: Optional[str] = None

    def prepare_engine(self):
        if "UNITXT_INFERENCE_ENGINE" in os.environ:
            engine_reference = os.environ["UNITXT_INFERENCE_ENGINE"]
        else:
            assert self.default is not None, (
                "GenericInferenceEngine could not be initialized"
                '\nThis is since both the "UNITXT_INFERENCE_ENGINE" environmental variable is not set and no default engine was not inputted.'
                "\nFor example, you can fix it by setting"
                "\nexport UNITXT_INFERENCE_ENGINE=engines.ibm_gen_ai.llama_3_70b_instruct"
                "\nto your ~/.bashrc"
                "\nor passing a similar required engine in the default argument"
            )
            engine_reference = self.default
        self.engine, _ = fetch_artifact(engine_reference)

    def get_engine_id(self):
        return "generic_inference_engine"

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        return self.engine._infer(dataset)


class OllamaInferenceEngine(InferenceEngine, PackageRequirementsMixin):
    label: str = "ollama"
    model_name: str
    _requirements_list = {
        "ollama": "Install ollama package using 'pip install --upgrade ollama"
    }
    data_classification_policy = ["public", "proprietary"]

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

    def prepare_engine(self):
        pass

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        import ollama

        result = [
            ollama.chat(
                model="llama2",
                messages=[
                    {
                        "role": "user",
                        "content": instance["source"],
                    },
                ],
            )
            for instance in dataset
        ]
        return [element["message"]["content"] for element in result]


class IbmGenAiInferenceEngine(
    InferenceEngine,
    IbmGenAiInferenceEngineParamsMixin,
    PackageRequirementsMixin,
    LogProbInferenceEngine,
):
    label: str = "ibm_genai"
    model_name: str
    _requirements_list = {
        "genai": "Install ibm-genai package using 'pip install --upgrade ibm-generative-ai"
    }
    data_classification_policy = ["public", "proprietary"]
    rate_limit: int = 10
    parameters: Optional[IbmGenAiInferenceEngineParams] = None

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

    @staticmethod
    def _get_credentials():
        from genai import Credentials

        api_key_env_var_name = "GENAI_KEY"
        api_key = os.environ.get(api_key_env_var_name)

        assert api_key is not None, (
            f"Error while trying to run IbmGenAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )

        return Credentials(api_key=api_key)

    def prepare_engine(self):
        self.check_missing_requirements()

        from genai import Client
        from genai.text.generation import CreateExecutionOptions

        credentials = self._get_credentials()
        self.client = Client(credentials=credentials)

        self.execution_options = CreateExecutionOptions(
            concurrency_limit=self.rate_limit
        )

        self._set_inference_parameters()

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        from genai.schema import TextGenerationParameters, TextGenerationResult

        genai_params = TextGenerationParameters(
            **self.to_dict([IbmGenAiInferenceEngineParamsMixin])
        )

        responses = self.client.text.generation.create(
            model_id=self.model_name,
            inputs=dataset["source"],
            parameters=genai_params,
            execution_options=self.execution_options,
        )

        results = []
        for response in responses:
            generation_result: TextGenerationResult = response.results[0]
            result = self.get_return_object(
                generation_result.generated_text, generation_result, return_meta_data
            )
            results.append(result)
        return results

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        from genai.schema import TextGenerationParameters, TextGenerationResult

        logprobs_return_options = {
            "generated_tokens": True,
            "input_text": False,
            "input_tokens": False,
            "token_logprobs": True,
            "token_ranks": True,
            "top_n_tokens": 5,
        }
        genai_params = self.to_dict(
            [IbmGenAiInferenceEngineParamsMixin], keep_empty=False
        )
        genai_params = {**genai_params, "return_options": logprobs_return_options}
        genai_params = TextGenerationParameters(**genai_params)
        predictions = self.client.text.generation.create(
            model_id=self.model_name,
            inputs=dataset["source"],
            parameters=genai_params,
            execution_options=self.execution_options,
        )

        predict_results = []
        for prediction in predictions:
            result: TextGenerationResult = prediction.results[0]
            assert isinstance(
                result.generated_tokens, list
            ), "result.generated_tokens should be a list"

            predict_result = []
            for base_token in result.generated_tokens:
                res = {**base_token.__dict__, **base_token.model_extra}
                res["top_tokens"] = [
                    {"logprob": top_token.logprob, "text": top_token.text}
                    for top_token in res["top_tokens"]
                ]
                predict_result.append(res)
            final_results = self.get_return_object(
                predict_result, result, return_meta_data
            )
            predict_results.append(final_results)
        return predict_results

    def get_return_object(self, predict_result, result, return_meta_data):
        if return_meta_data:
            return TextGenerationInferenceOutput(
                prediction=predict_result,
                input_tokens=result.input_token_count,
                output_tokens=result.generated_token_count,
                model_name=self.model_name,
                inference_type=self.label,
                input_text=result.input_text,
                seed=self.random_seed,
                stop_reason=result.stop_reason,
            )
        return predict_result

    def get_model_details(self) -> Dict:
        from genai import ApiClient
        from genai.model import ModelService

        api_client = ApiClient(credentials=self._get_credentials())
        model_info = (
            ModelService(api_client=api_client).retrieve(id=self.model_name).result
        )
        return model_info.dict()


class OpenAiInferenceEngineParamsMixin(Artifact):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = 20
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = True
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    service_tier: Optional[Literal["auto", "default"]] = None


@deprecation(version="2.0.0", alternative=OpenAiInferenceEngineParamsMixin)
class OpenAiInferenceEngineParams(Artifact):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = 20
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = True
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    service_tier: Optional[Literal["auto", "default"]] = None


class OpenAiInferenceEngine(
    InferenceEngine,
    LogProbInferenceEngine,
    OpenAiInferenceEngineParamsMixin,
    PackageRequirementsMixin,
):
    label: str = "openai"
    model_name: str
    _requirements_list = {
        "openai": "Install openai package using 'pip install --upgrade openai"
    }
    data_classification_policy = ["public"]
    parameters: Optional[OpenAiInferenceEngineParams] = None

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

    @classmethod
    def get_api_param(cls, inference_engine: str, api_param_env_var_name: str):
        api_key = os.environ.get(api_param_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run {inference_engine}."
            f" Please set the environment param '{api_param_env_var_name}'."
        )
        return api_key

    def create_client(self):
        from openai import OpenAI

        api_key = self.get_api_param(
            inference_engine="OpenAiInferenceEngine",
            api_param_env_var_name="OPENAI_API_KEY",
        )
        return OpenAI(api_key=api_key)

    def prepare_engine(self):
        self.client = self.create_client()
        self._set_inference_parameters()

    def _get_completion_kwargs(self):
        return {
            k: v
            for k, v in self.to_dict([OpenAiInferenceEngineParamsMixin]).items()
            if v is not None
        }

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        outputs = []
        for instance in tqdm(dataset, desc="Inferring with openAI API"):
            response = self.client.chat.completions.create(
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": self.system_prompt,
                    # },
                    {
                        "role": "user",
                        "content": instance["source"],
                    }
                ],
                model=self.model_name,
                **self._get_completion_kwargs(),
            )
            prediction = response.choices[0].message.content
            output = self.get_return_object(prediction, response, return_meta_data)

            outputs.append(output)

        return outputs

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        outputs = []
        for instance in tqdm(dataset, desc="Inferring with openAI API"):
            response = self.client.chat.completions.create(
                messages=[
                    # {
                    #     "role": "system",
                    #     "content": self.system_prompt,
                    # },
                    {
                        "role": "user",
                        "content": instance["source"],
                    }
                ],
                model=self.model_name,
                **self._get_completion_kwargs(),
            )
            top_logprobs_response = response.choices[0].logprobs.content
            pred_output = [
                {
                    "top_tokens": [
                        {"text": obj.token, "logprob": obj.logprob}
                        for obj in generated_token.top_logprobs
                    ]
                }
                for generated_token in top_logprobs_response
            ]
            output = self.get_return_object(pred_output, response, return_meta_data)
            outputs.append(output)
        return outputs

    def get_return_object(self, predict_result, response, return_meta_data):
        if return_meta_data:
            return TextGenerationInferenceOutput(
                prediction=predict_result,
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                model_name=self.model_name,
                inference_type=self.label,
            )
        return predict_result


class TogetherAiInferenceEngineParamsMixin(Artifact):
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    logprobs: Optional[int] = None
    echo: Optional[bool] = None
    n: Optional[int] = None
    min_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None


class TogetherAiInferenceEngine(
    InferenceEngine, TogetherAiInferenceEngineParamsMixin, PackageRequirementsMixin
):
    label: str = "together"
    model_name: str
    _requirements_list = {
        "together": "Install together package using 'pip install --upgrade together"
    }
    data_classification_policy = ["public"]
    parameters: Optional[TogetherAiInferenceEngineParamsMixin] = None

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, self.label)

    def prepare_engine(self):
        from together import Together
        from together.types.models import ModelType

        api_key_env_var_name = "TOGETHER_API_KEY"
        api_key = os.environ.get(api_key_env_var_name)
        assert api_key is not None, (
            f"Error while trying to run TogetherAiInferenceEngine."
            f" Please set the environment param '{api_key_env_var_name}'."
        )
        self.client = Together(api_key=api_key)
        self._set_inference_parameters()

        # Get model type from Together List Models API
        together_models = self.client.models.list()
        together_model_id_to_type = {
            together_model.id: together_model.type for together_model in together_models
        }
        model_type = together_model_id_to_type.get(self.model_name)
        assert model_type is not None, (
            f"Could not find model {self.model_name} " "in Together AI model list"
        )
        assert model_type in [ModelType.CHAT, ModelType.LANGUAGE, ModelType.CODE], (
            f"Together AI model type {model_type} is not supported; "
            "supported types are 'chat', 'language' and 'code'."
        )
        self.model_type = model_type

    def _get_infer_kwargs(self):
        return {
            k: v
            for k, v in self.to_dict([TogetherAiInferenceEngineParamsMixin]).items()
            if v is not None
        }

    def _infer_chat(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **self._get_infer_kwargs(),
        )
        return response.choices[0].message.content

    def _infer_text(self, prompt: str) -> str:
        response = self.client.completions.create(
            model=self.model_name,
            prompt=prompt,
            **self._get_infer_kwargs(),
        )
        return response.choices[0].text

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        from together.types.models import ModelType

        outputs = []
        if self.model_type == ModelType.CHAT:
            for instance in tqdm(dataset, desc="Inferring with Together AI Chat API"):
                outputs.append(self._infer_chat(instance["source"]))
        else:
            for instance in tqdm(dataset, desc="Inferring with Together AI Text API"):
                outputs.append(self._infer_text(instance["source"]))
        return outputs


class VLLMRemoteInferenceEngine(OpenAiInferenceEngine):
    label: str = "vllm"

    def create_client(self):
        from openai import OpenAI

        api_key = self.get_api_param(
            inference_engine="VLLMRemoteInferenceEngine",
            api_param_env_var_name="VLLM_API_KEY",
        )
        api_url = self.get_api_param(
            inference_engine="VLLMRemoteInferenceEngine",
            api_param_env_var_name="VLLM_API_URL",
        )
        return OpenAI(api_key=api_key, base_url=api_url)


class WMLInferenceEngineParamsMixin(Artifact):
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    length_penalty: Optional[Dict[str, Union[int, float]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    min_new_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    time_limit: Optional[int] = None
    truncate_input_tokens: Optional[int] = None
    prompt_variables: Optional[Dict[str, Any]] = None
    return_options: Optional[Dict[str, bool]] = None


@deprecation(version="2.0.0", alternative=WMLInferenceEngineParamsMixin)
class WMLInferenceEngineParams(Artifact):
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    length_penalty: Optional[Dict[str, Union[int, float]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    min_new_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    time_limit: Optional[int] = None
    truncate_input_tokens: Optional[int] = None
    prompt_variables: Optional[Dict[str, Any]] = None
    return_options: Optional[Dict[str, bool]] = None


CredentialsWML = Dict[
    Literal["url", "username", "password", "apikey", "project_id", "space_id"], str
]


class WMLInferenceEngine(
    InferenceEngine,
    WMLInferenceEngineParamsMixin,
    PackageRequirementsMixin,
    LogProbInferenceEngine,
):
    """Runs inference using ibm-watsonx-ai.

    Attributes:
        credentials (Dict[str, str], optional): By default, it is created by a class
            instance which tries to retrieve proper environment variables
            ("WML_URL", "WML_PROJECT_ID", "WML_SPACE_ID", "WML_APIKEY", "WML_USERNAME", "WML_PASSWORD").
            However, a dictionary with the following keys: "url", "apikey", "project_id", "space_id",
            "username", "password".
            can be directly provided instead.
        model_name (str, optional): ID of a model to be used for inference. Mutually
            exclusive with 'deployment_id'.
        deployment_id (str, optional): Deployment ID of a tuned model to be used for
            inference. Mutually exclusive with 'model_name'.
        parameters (WMLInferenceEngineParams, optional): Instance of WMLInferenceEngineParams
            which defines inference parameters and their values. Deprecated attribute, please
            pass respective parameters directly to the WMLInferenceEngine class instead.
        concurrency_limit (int): number of requests that will be sent in parallel, max is 10.

    Examples:
        from .api import load_dataset

        wml_credentials = {
            "url": "some_url", "project_id": "some_id", "api_key": "some_key"
        }
        model_name = "google/flan-t5-xxl"
        wml_inference = WMLInferenceEngine(
            credentials=wml_credentials,
            model_name=model_name,
            data_classification_policy=["public"],
            top_p=0.5,
            random_seed=123,
        )

        dataset = load_dataset(
            dataset_query="card=cards.argument_topic,template_card_index=0,loader_limit=5"
        )
        results = wml_inference.infer(dataset["test"])
    """

    credentials: Optional[CredentialsWML] = None
    model_name: Optional[str] = None
    deployment_id: Optional[str] = None
    label: str = "wml"
    _requirements_list = {
        "ibm_watsonx_ai": "Install ibm-watsonx-ai package using 'pip install --upgrade ibm-watsonx-ai'. "
        "It is advised to have Python version >=3.10 installed, as at lower version this package "
        "may cause conflicts with other installed packages."
    }
    data_classification_policy = ["public", "proprietary"]
    parameters: Optional[WMLInferenceEngineParams] = None
    concurrency_limit: int = 10
    _client: Any = InternalField(default=None, name="WML client")
    _model: Any = InternalField(default=None, name="WML model")

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name or self.deployment_id, self.label)

    def verify(self):
        super().verify()

        assert (
            isinstance(self.concurrency_limit, int)
            and 1 <= self.concurrency_limit <= 10
        ), (
            f"'concurrency_limit' must be a positive integer not greater than 10. "
            f"However, '{self.concurrency_limit}' was given."
        )

        assert (
            self.model_name
            or self.deployment_id
            and not (self.model_name and self.deployment_id)
        ), "Either 'model_name' or 'deployment_id' must be specified, but not both at the same time."

    def process_data_before_dump(self, data):
        if "credentials" in data:
            for key, value in data["credentials"].items():
                if key != "url":
                    data["credentials"][key] = "<hidden>"
                else:
                    data["credentials"][key] = value
        return data

    def _initialize_wml_client(self):
        from ibm_watsonx_ai.client import APIClient

        if self.credentials is None:
            self.credentials = self._read_wml_credentials_from_env()
        self._verify_wml_credentials(self.credentials)

        client = APIClient(credentials=self.credentials)
        if "space_id" in self.credentials:
            client.set.default_space(self.credentials["space_id"])
        else:
            client.set.default_project(self.credentials["project_id"])
        return client

    @staticmethod
    def _read_wml_credentials_from_env() -> CredentialsWML:
        credentials: CredentialsWML = {}

        url = os.environ.get("WML_URL")
        assert url, (
            "Error while trying to run 'WMLInferenceEngine'. "
            "Please set the env variable: 'WML_URL'"
        )
        credentials["url"] = url

        space_id = os.environ.get("WML_SPACE_ID")
        project_id = os.environ.get("WML_PROJECT_ID")
        if space_id and project_id:
            get_logger().warning(
                "Either 'WML_SPACE_ID' or 'WML_PROJECT_ID' need to be "
                "specified, however, both were found. 'WMLInferenceEngine' "
                "will use space by default. If it is not desired, then have "
                "only one of those defined in the env."
            )
            credentials["space_id"] = space_id
        elif project_id:
            credentials["project_id"] = project_id
        else:
            raise AssertionError(
                "Error while trying to run 'WMLInferenceEngine'. "
                "Please set either 'WML_SPACE_ID' or 'WML_PROJECT_ID' env "
                "variable."
            )

        apikey = os.environ.get("WML_APIKEY")
        username = os.environ.get("WML_USERNAME")
        password = os.environ.get("WML_PASSWORD")

        if apikey and username and password:
            get_logger().warning(
                "Either 'WML_APIKEY' or both 'WML_USERNAME' and 'WML_PASSWORD' "
                "need to be specified, however, all of them were found. "
                "'WMLInferenceEngine' will use api key only by default. If it is not "
                "desired, then have only one of those options defined in the env."
            )

        if apikey:
            credentials["apikey"] = apikey
        elif username and password:
            credentials["username"] = username
            credentials["password"] = password
        else:
            raise AssertionError(
                "Error while trying to run 'WMLInferenceEngine'. "
                "Please set either 'WML_APIKEY' or both 'WML_USERNAME' and "
                "'WML_PASSWORD' env variables."
            )

        return credentials

    @staticmethod
    def _verify_wml_credentials(credentials: CredentialsWML) -> None:
        assert isoftype(credentials, CredentialsWML), (
            "WML credentials object must be a dictionary which may "
            "contain only the following keys: "
            "['url', 'apikey', 'username', 'password']."
        )

        assert credentials.get(
            "url"
        ), "'url' is a mandatory key for WML credentials dict."
        assert "space_id" in credentials or "project_id" in credentials, (
            "Either 'space_id' or 'project_id' must be provided "
            "as keys for WML credentials dict."
        )
        assert "apikey" in credentials or (
            "username" in credentials and "password" in credentials
        ), (
            "Either 'apikey' or both 'username' and 'password' must be provided "
            "as keys for WML credentials dict."
        )

    def prepare_engine(self):
        self.check_missing_requirements()

        self._client = self._initialize_wml_client()

        self._set_inference_parameters()

    def _load_model(self):
        from ibm_watsonx_ai.foundation_models.inference import ModelInference

        self._model = ModelInference(
            model_id=self.model_name,
            deployment_id=self.deployment_id,
            api_client=self._client,
        )

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if self._model is None:
            self._load_model()

        params = self.to_dict([WMLInferenceEngineParamsMixin], keep_empty=False)

        return self._send_requests(
            params=params,
            inputs=dataset["source"],
            return_logprobs=False,
            return_meta_data=return_meta_data,
        )

    def _infer_log_probs(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[Dict], List[TextGenerationInferenceOutput]]:
        if self._model is None:
            self._load_model()

        params = self.to_dict([WMLInferenceEngineParamsMixin], keep_empty=False)

        user_return_options = params.pop("return_options", {})
        # currently this is the only configuration that returns generated logprobs and behaves as expected
        logprobs_return_options = {
            "input_tokens": True,
            "generated_tokens": True,
            "token_logprobs": True,
            "top_n_tokens": user_return_options.get("top_n_tokens", 5),
        }
        for key, value in logprobs_return_options.items():
            if key in user_return_options and user_return_options[key] != value:
                raise ValueError(
                    f"'{key}={user_return_options[key]}' is not supported for the 'infer_log_probs' "
                    f"method of {self.__class__.__name__}. For obtaining the logprobs of generated tokens "
                    f"please use '{key}={value}'."
                )

        params = {
            **params,
            "return_options": logprobs_return_options,
        }

        return self._send_requests(
            params=params,
            inputs=dataset["source"],
            return_logprobs=True,
            return_meta_data=return_meta_data,
        )

    def _send_requests(
        self,
        params: Dict[str, Any],
        inputs: List[str],
        return_logprobs: bool,
        return_meta_data: bool = False,
    ) -> Union[List[str], List[Dict], List[TextGenerationInferenceOutput]]:
        generation_type = "generated_tokens" if return_logprobs else "generated_text"

        results = self._model.generate(
            prompt=inputs,
            params=params,
            concurrency_limit=self.concurrency_limit,
        )

        final_results = []
        for result, inp in zip(results, inputs):
            result_metadata = result["results"][0]
            generated_content = result_metadata[generation_type]
            final_results.append(
                self.get_return_object(
                    generated_content, result_metadata, inp, return_meta_data
                )
            )
        return final_results

    def get_return_object(self, predict_result, result, input_text, return_meta_data):
        if return_meta_data:
            return TextGenerationInferenceOutput(
                prediction=predict_result,
                input_tokens=result["input_token_count"],
                output_tokens=result["generated_token_count"],
                model_name=self.model_name or self.deployment_id,
                inference_type=self.label,
                stop_reason=result["stop_reason"],
                seed=self.random_seed,
                input_text=input_text,
            )
        return predict_result

    def get_model_details(self) -> Dict:
        return self._model.get_details()


class HFLlavaInferenceEngine(InferenceEngine, LazyLoadMixin):
    model_name: str
    max_new_tokens: int
    lazy_load = True

    _requirements_list = {
        "transformers": "Install huggingface package using 'pip install --upgrade transformers",
        "torch": "Install torch, go on PyTorch website for mode details.",
        "accelerate": "pip install accelerate",
    }

    def get_engine_id(self):
        return get_model_and_label_id(self.model_name, "hf_lava")

    def _prepare_engine(self):
        import torch
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.device = torch.device(
            "mps"
            if torch.backends.mps.is_available()
            else 0
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(self.model_name)

    def prepare_engine(self):
        if not self.lazy_load:
            self._prepare_engine()

    def _is_loaded(self):
        return hasattr(self, "model") and self.model is not None

    def _infer(
        self,
        dataset: Union[List[Dict[str, Any]], DatasetDict],
        return_meta_data: bool = False,
    ) -> Union[List[str], List[TextGenerationInferenceOutput]]:
        if not self._is_loaded():
            self._prepare_engine()

        import torch

        results = []
        for instance in tqdm(dataset):
            text = instance["source"]
            images = extract_images(text, instance)
            # Regular expression to match all <img src="..."> tags
            regex = r'<img\s+src=["\'](.*?)["\']\s*/?>'
            model_input = re.sub(regex, "<image>", text)
            if len(images) == 1:
                images = images[0]
            inputs = self.processor(
                images=images, text=model_input, return_tensors="pt"
            ).to(self.device, torch.float16)
            input_len = len(inputs["input_ids"][0])
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            result = self.processor.decode(
                output[0][input_len:], skip_special_tokens=True
            )
            results.append(result)

        return results
