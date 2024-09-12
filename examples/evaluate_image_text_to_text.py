from unitxt.api import evaluate, load_dataset
from unitxt.inference import HFLlavaInferenceEngine
from unitxt.text_utils import print_dict
from cvar_pyutils.debugging_tools import set_remote_debugger
set_remote_debugger('9.61.126.161', 55557)

inference_model = HFLlavaInferenceEngine(
    model_name="llava-hf/llava-interleave-qwen-0.5b-hf", max_new_tokens=32
)

dataset = load_dataset(
    card="cards.doc_vqa.en",
    template="templates.qa.with_context.title",
    format="formats.models.llava_interleave",
    # format="formats.models.phi_3_MM",
    loader_limit=20,
    # augmentor="augmentors.white_space",
)

test_dataset = dataset["test"].select(range(5))

predictions = inference_model.infer(test_dataset)
evaluated_dataset = evaluate(predictions=predictions, data=test_dataset)

print_dict(
    evaluated_dataset[0],
    keys_to_print=["source", "media", "references", "processed_prediction", "score"],
)
