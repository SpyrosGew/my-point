from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from PIL import Image


def parse_json(output: str) -> dict:
    match = re.search(r"\{.*\}", output, flags=re.DOTALL)
    if not match:
        raise RuntimeError(f"VLM did not return JSON: {output[:300]}")
    data = json.loads(match.group(0))
    return {
        "target_id": str(data.get("target_id", "")).strip().upper(),
        "confidence": float(data.get("confidence", 0.0)),
        "reason": str(data.get("reason", ""))[:240],
    }


def generated_text_from_pipeline_output(output: object) -> str:
    if isinstance(output, list) and output:
        return generated_text_from_pipeline_output(output[0])
    if isinstance(output, dict):
        value = output.get("generated_text", "")
        if isinstance(value, list) and value:
            last = value[-1]
            if isinstance(last, dict):
                content = last.get("content", "")
                if isinstance(content, list):
                    return " ".join(str(item.get("text", "")) for item in content if isinstance(item, dict))
                return str(content)
        return str(value)
    return str(output)


def main() -> None:
    if len(sys.argv) != 2:
        raise RuntimeError("usage: python -m vlm_select outputs/vlm_request.json")

    request = json.loads(Path(sys.argv[1]).read_text())
    image = Image.open(request["image_path"]).convert("RGB")
    crop_sheet_path = request.get("crop_sheet_path")
    crop_sheet = Image.open(crop_sheet_path).convert("RGB") if crop_sheet_path else None
    group_crop_path = request.get("group_crop_path")
    group_crop = Image.open(group_crop_path).convert("RGB") if group_crop_path else None
    command = request["command"]
    candidates = request["candidates"]
    target_class = request.get("target_class")
    model_name = request.get("model") or os.getenv("VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    device = request.get("device") or os.getenv("VLM_DEVICE", "cuda")
    max_new_tokens = int(request.get("max_new_tokens") or os.getenv("VLM_MAX_NEW_TOKENS", "96"))

    import torch

    backend = os.getenv("VLM_BACKEND", "qwen").strip().lower()
    load_in_4bit = os.getenv("VLM_LOAD_IN_4BIT", "0").lower() in {"1", "true", "yes"}
    dtype = torch.bfloat16 if torch.cuda.is_available() and device == "cuda" else torch.float32
    model = None
    processor = None
    pipe = None
    if backend == "internvl":
        from transformers import pipeline

        pipe_kwargs = {"trust_remote_code": True}
        if device == "cuda":
            pipe_kwargs["device_map"] = "auto"
            pipe_kwargs["torch_dtype"] = dtype
        pipe = pipeline("image-text-to-text", model=model_name, **pipe_kwargs)
    else:
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        model_kwargs = {"torch_dtype": dtype}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            if load_in_4bit:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type=os.getenv("VLM_BNB_4BIT_TYPE", "nf4"),
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, **model_kwargs)
        if device != "cuda":
            model.to(device)
        processor = AutoProcessor.from_pretrained(model_name)

    candidates_text = "\n".join(
        f"{item['id']}: {item['label']} position_among_same_class={item.get('same_class_position', 'unknown')} "
        f"depth_position_among_same_class={item.get('same_class_depth_position', 'unknown')} "
        f"depth_m={item.get('depth_m')} center={item.get('center')} box={item['box']} pointing_score={item['pointing_score']:.2f} "
        f"spatial_score={item['language_score']:.2f}"
        for item in candidates
    )
    prompt = (
        "You are selecting the object a robot should fetch. The image has a yellow pointing line "
        "and candidate objects marked with letter IDs. The detector has already narrowed the scene "
        "using the user's pointing ray and target object class. Choose only from the labeled candidates. "
        "Objects mentioned as landmarks or relations are context, not the target. "
        "Use the candidate group crop as the main evidence, because it shows "
        "all candidate objects together in one enlarged view. The full scene or crop sheet may be omitted for speed. Use any crop sheet as backup for fine visual details "
        "such as scarf, color, texture, objects on the chair, or other attributes. "
        "For positional language, reason within the required target class and within the group crop: left/right/middle/center "
        "use relative image position among same-class candidates, top/bottom use vertical position, and back/behind/far/front/near "
        "use depth_m plus visible scene layout. Metadata is a strong hint, but choose the object that best matches the complete command. "
        "If the command describes a visual attribute, inspect both the group crop and individual crops carefully.\n"
        f"User command: {command!r}\n"
        f"Required target class: {target_class or 'unknown'}\n"
        f"Candidates:\n{candidates_text}\n"
        "Return strict JSON only: {\"target_id\":\"A\",\"confidence\":0.0,"
        "\"reason\":\"short reason\"}. Never choose a landmark object if it is not the required target class."
    )
    send_full_scene = os.getenv("VLM_SEND_FULL_SCENE", "1").lower() in {"1", "true", "yes"}
    send_crop_sheet = os.getenv("VLM_SEND_CROP_SHEET", "1").lower() in {"1", "true", "yes"}
    content = []
    if send_full_scene or group_crop is None:
        content.append({"type": "image", "image": image})
    if group_crop is not None:
        content.append({"type": "image", "image": group_crop})
    if send_crop_sheet and crop_sheet is not None:
        content.append({"type": "image", "image": crop_sheet})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if backend == "internvl":
        output = pipe(text=messages, max_new_tokens=max_new_tokens, return_full_text=False)
        print(json.dumps(parse_json(generated_text_from_pipeline_output(output))))
        return

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print(json.dumps(parse_json(output)))


if __name__ == "__main__":
    main()
