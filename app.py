from dotenv import load_dotenv
load_dotenv()
import gradio as gr
import nest_asyncio
import asyncio
from openai import OpenAIError, AsyncOpenAI
import tiktoken
import re
import json
import os

from lightrag import LightRAG, QueryParam
from lightrag.utils import setup_logger, EmbeddingFunc
from lightrag.llm.openai import openai_embed

# === Setup ===
nest_asyncio.apply()
setup_logger("lighrag", level="INFO")

WORKING_DIR = os.environ.get("WORKING_DIR", "./rag_storage")
client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
manual_token_usage = {"call_count": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

def count_tokens(text, model="gpt-4o"):
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

TOXIC_KEYWORDS = ["hate", "kill", "stupid", "idiot", "dumb"]
HALLUCINATION_PATTERNS = [
    r"\bAs an AI\b", r"\bI assume\b", r"\bnot mentioned\b", r"\bno data\b", r"\bI do not have access\b"
]

def validate_response(response):
    issues = []
    for word in TOXIC_KEYWORDS:
        if re.search(rf"\b{word}\b", response, re.IGNORECASE):
            issues.append(f"Toxic word: {word}")
    for pattern in HALLUCINATION_PATTERNS:
        if re.search(pattern, response, re.IGNORECASE):
            issues.append(f"Hallucination pattern: {pattern}")
    return {"is_safe": len(issues) == 0, "issues": issues}

async def tracked_llm_model_func(prompt, system_prompt=None, temperature=0.2, max_tokens=512):
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": prompt})
    try:
        response = await client.chat.completions.create(
            model="gpt-4o", messages=messages, temperature=temperature, max_tokens=max_tokens
        )
    except OpenAIError as e:
        return "[ERROR] LLM call failed: " + str(e)

    reply = response.choices[0].message.content
    validation = validate_response(reply)
    if not validation["is_safe"]:
        reply = "[BLOCKED] ⚠️ Response flagged: " + ", ".join(validation["issues"])

    prompt_tokens = count_tokens(system_prompt + prompt if system_prompt else prompt)
    completion_tokens = count_tokens(reply)
    manual_token_usage["call_count"] += 1
    manual_token_usage["prompt_tokens"] += prompt_tokens
    manual_token_usage["completion_tokens"] += completion_tokens
    manual_token_usage["total_tokens"] += prompt_tokens + completion_tokens

    return reply

async def init_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=tracked_llm_model_func,
        embedding_func=EmbeddingFunc(embedding_dim=1536, max_token_size=8192, func=openai_embed),
        enable_llm_cache=False,
        enable_llm_cache_for_entity_extract=False
    )
    await rag.initialize_storages()
    return rag

async def query_pipeline(query, mode="mix"):
    rag = await init_rag()
    result = rag.query(
        query,
        param=QueryParam(mode=mode, model_func=tracked_llm_model_func)
    )

    if "### References" in result:
        answer_part, ref_part = result.split("### References", 1)
    else:
        answer_part, ref_part = result, ""

    if not ref_part.strip() or "[General Knowledge]" in ref_part:
        answer_part = "⚠️ The answer could not be verified with supporting references from the knowledge base."
        ref_part = "No references found. The model may have generated the answer without grounded context."

    return answer_part.strip(), ref_part.strip(), json.dumps(manual_token_usage, indent=2)

def run_gradio(query):
    return asyncio.run(query_pipeline(query))

# === Gradio UI ===
with gr.Blocks() as demo:
    gr.Markdown("## 🧑‍💻 ElectronicGPT")

    with gr.Row():
        query_input = gr.Textbox(label="Ask your question", lines=2, placeholder="E.g., 'What is soldering?'")

    run_btn = gr.Button("🔍 Run Query")

    answer_output = gr.Textbox(label="🧠 Answer", lines=8)
    reference_output = gr.Textbox(label="📋 References", lines=3)
    usage_output = gr.Textbox(label="🗞️ Token Usage", lines=5)

    run_btn.click(fn=run_gradio, inputs=[query_input], outputs=[answer_output, reference_output, usage_output])

    demo.launch(server_name="0.0.0.0", server_port=7860)
