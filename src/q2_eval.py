"""
Q2 evaluation pipeline.

Mirrors `eval_v2.py` (Q1) but for Q2 multiple-choice cubemap MCQ:
  - Loads pre-rendered cubemap from data/setA_extended_q2/{strategy}/images/
  - Builds prompt aligned with Q1 sighted prompt
  - Two image-input variants:
      cubemap_only        — 1 image (just the cubemap)
      frame_plus_cubemap  — 2 images (original frame + cubemap)
  - Two distractor strategies × two input variants = 4 conditions
  - Sends to vLLM endpoint, parses {"answer": "A|B|C|D"} from JSON

Output: one JSONL per (condition, model):
  meta/testset/exp/q2_eval_001/runs/{condition}_{model}_M{M}.jsonl
  Each line: {sample_id, repeat_id, q2_strategy, input_variant,
              gt_letter, gt_face, raw, parsed}

Run:
  conda run -n slam python3 -m src.q2_eval                   # all
  conda run -n slam python3 -m src.q2_eval --max 50          # smoke test
  conda run -n slam python3 -m src.q2_eval --conditions A_cubemap_only \
      --models qwen3.5-9b
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")

# ── Paths ─────────────────────────────────────────────────────────────
# Env-driven so a fresh clone + `huggingface-cli download
# nipsedtrack2026/q2-cubemap-mcq --local-dir data/q2` works out of the
# box from the eval-code root.
TESTSET = Path(os.environ.get(
    "TESTSET_ROOT",
    str(Path(__file__).resolve().parents[1]),
))
Q2_DATASET_ROOT = Path(os.environ.get(
    "Q2_DATASET_ROOT",
    str(TESTSET / "data" / "q2"),
))
Q1_DATASET_ROOT = Path(os.environ.get(
    "Q1_DATASET_ROOT",
    str(TESTSET / "data" / "q1"),
))
DATA_DIR = TESTSET / "data"
RUNS_DIR = Path(os.environ.get(
    "Q2_RUNS_DIR",
    str(TESTSET / "runs" / "q2"),
))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

EK_ROOT = Path(os.environ.get(
    "EK_FRAMES_ROOT",
    "/path/to/epic-kitchens",
))
HD_ROOT = Path(os.environ.get(
    "HD_EPIC_FRAMES_ROOT",
    "/path/to/hd-epic/Participants",
))

# ── Model endpoints (same as eval_v2.py) ─────────────────────────────
MODELS = {
    'qwen3.5-9b': {
        'api': 'openai_compat',
        'base_url': 'http://127.0.0.1:8001/v1/chat/completions',
        'model_id': 'Qwen/Qwen3.5-9B',
        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}},
    },
    'gemma-4-31b': {
        'api': 'openai_compat',
        'base_url': 'http://127.0.0.1:9002/v1/chat/completions',
        'model_id': 'google/gemma-4-31b-it',
        'extra_body': {},
    },
    'qwen3-vl-30b': {
        'api': 'openai_compat',
        'base_url': 'http://127.0.0.1:18002/v1/chat/completions',
        'model_id': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'extra_body': {},
    },
    'gemini-3-flash': {
        'api': 'gemini',
        'model_id': 'gemini-3-flash-preview',
        'env_key': 'GEMINI_API_KEY',
    },
    'gpt-5.4': {
        'api': 'openai',
        'model_id': 'gpt-5.4',
        'env_key': 'OPENAI_API_KEY',
    },
}
API_CONCURRENCY = 4  # lower limit for paid APIs to be polite to rate limits

# ── Eval config ──────────────────────────────────────────────────────
# Q2 is multiple-choice with a sharp GT — single sample per query is enough
# for argmax accuracy. (Q1 needs M=20 because its answer space is a
# distribution over bins.)
M_REPEATS = 1
TEMPERATURE = 1.0
MAX_TOKENS = 256        # local models — JSON is short
MAX_TOKENS_API = 4096   # API "thinking" models (Gemini-3, GPT-5.4) burn many tokens internally before emitting JSON
CONCURRENCY_PER_MODEL = 8
SYSTEM_PROMPT = "You are a careful spatial reasoner."

# Conditions = (distractor_strategy, input_variant)
STRATEGIES = ['A_random_diff_face', 'C_no_target_cluster']
INPUT_VARIANTS = ['cubemap_only', 'frame_plus_cubemap']


def list_conditions():
    out = []
    for s in STRATEGIES:
        s_short = s.split('_')[0]  # 'A' or 'C'
        for v in INPUT_VARIANTS:
            out.append(f'{s_short}_{v}')
    return out


CONDITIONS = list_conditions()


def parse_condition(cond_name):
    """'A_cubemap_only' → ('A_random_diff_face', 'cubemap_only')."""
    parts = cond_name.split('_', 1)
    s_short, variant = parts[0], parts[1]
    strat = next(s for s in STRATEGIES if s.startswith(s_short + '_'))
    return strat, variant


# ── Prompt builders (aligned with eval_v2.py:build_sighted_prompt) ───
LAYOUT_DESC = (
    "The cubemap is a 2x3 grid showing every direction from the camera:\n"
    "  top row:    UP   | FRONT  | DOWN\n"
    "  bottom row: LEFT | BEHIND | RIGHT\n"
    "The dark-grey region in each tile is OUTSIDE the camera's original field "
    "of view (not seen by the camera). The visible RGB region shows what the "
    "camera actually saw."
)

OPTIONS_DESC = (
    "Four candidate locations are marked in the unseen (dark-grey) region "
    "with coloured letter markers: A, B, C, and D. Exactly one marker is at "
    "the true 3D location of the target object; the other three are distractors."
)


def build_prompt_cubemap_only(target: str) -> str:
    return f"""\
You are given ONE egocentric RGB image (first-person view), projected as a 2x3 cubemap from the camera's viewpoint.
The target object "{target}" is OUT OF VIEW (not visible in any region of the cubemap).
Predict its most likely direction based on visible layout cues \
(walls, counters, appliances, doorways, free space) and spatial commonsense.

{LAYOUT_DESC}

{OPTIONS_DESC}

Output a single JSON object:
  {{"justification": "<1-2 sentences of spatial reasoning>",
    "answer": "<A|B|C|D>"}}

Example:
  {{"justification": "The sink is typically behind and to the left in this layout, and marker C is in that area.",
    "answer": "C"}}
"""


def build_prompt_frame_plus_cubemap(target: str) -> str:
    return f"""\
You are given TWO images: (1) the original egocentric RGB image (first-person view), and (2) the same image projected as a 2x3 cubemap from the camera's viewpoint.
The target object "{target}" is OUT OF VIEW (not visible in either image).
Predict its most likely direction based on visible layout cues \
(walls, counters, appliances, doorways, free space) and spatial commonsense.

{LAYOUT_DESC}

{OPTIONS_DESC}

Output a single JSON object:
  {{"justification": "<1-2 sentences of spatial reasoning>",
    "answer": "<A|B|C|D>"}}

Example:
  {{"justification": "The sink is typically behind and to the left in this layout, and marker C is in that area.",
    "answer": "C"}}
"""


PROMPT_BUILDERS = {
    'cubemap_only':       build_prompt_cubemap_only,
    'frame_plus_cubemap': build_prompt_frame_plus_cubemap,
}


# ── Frame I/O ────────────────────────────────────────────────────────
def resolve_frame_path(dataset: str, video_id: str, frame_index: int,
                       participant_id: str,
                       bundled_frame_path: str | None = None,
                       q2_strategy: str | None = None) -> str:
    """Resolve a source-frame path. Prefers `bundled_frame_path` (the
    relative path shipped by the HF q2-cubemap-mcq dataset under each
    strategy's `frames/` tree). Falls back to the legacy upstream layout.

    `q2_strategy` is required when using `bundled_frame_path` because
    each strategy ships its own `frames/` subtree under
    `Q2_DATASET_ROOT/<strategy>/`.
    """
    if bundled_frame_path:
        if q2_strategy:
            return str(Q2_DATASET_ROOT / q2_strategy / bundled_frame_path)
        # backward-compat: also try Q1's bundled location
        return str(Q1_DATASET_ROOT / bundled_frame_path)
    if dataset == 'epic_kitchens':
        return str(EK_ROOT / participant_id / video_id / 'frames'
                   / f'frame_{frame_index:010d}.jpg')
    else:
        return str(HD_ROOT / participant_id / video_id / 'images'
                   / f'frame_{frame_index:06d}.jpg')


def load_jpeg_bytes(path: str) -> bytes:
    with open(path, 'rb') as f:
        return f.read()


# ── VLM call (dispatches by API type) ────────────────────────────────
async def call_openai_compat(prompt, image_bytes_list, model_cfg,
                              temperature, max_tokens):
    import httpx
    content = []
    for ib in image_bytes_list:
        b64 = base64.b64encode(ib).decode('ascii')
        content.append({'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{b64}'}})
    content.append({'type': 'text', 'text': prompt})
    payload = {
        'model': model_cfg['model_id'],
        'messages': [
            {'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM_PROMPT}]},
            {'role': 'user', 'content': content},
        ],
        'temperature': temperature,
        'max_tokens': max_tokens,
    }
    if model_cfg.get('extra_body'):
        payload.update(model_cfg['extra_body'])
    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(5):
            try:
                resp = await client.post(
                    model_cfg['base_url'],
                    headers={'Authorization': 'Bearer placeholder'},
                    json=payload)
                if resp.status_code == 200:
                    return resp.json()['choices'][0]['message']['content']
                if resp.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f'ERROR: HTTP {resp.status_code}: {resp.text[:200]}'
            except Exception as e:
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f'ERROR: {e}'
    return 'ERROR: max retries'


async def call_gemini(prompt, image_bytes_list, model_cfg,
                       temperature, max_tokens):
    import os, httpx
    api_key = os.environ.get(model_cfg['env_key'])
    if not api_key:
        return f'ERROR: {model_cfg["env_key"]} not set'
    parts = []
    for ib in image_bytes_list:
        b64 = base64.b64encode(ib).decode('ascii')
        parts.append({'inlineData': {'mimeType': 'image/jpeg', 'data': b64}})
    parts.append({'text': prompt})
    payload = {
        'contents': [{'parts': parts}],
        'systemInstruction': {'parts': [{'text': SYSTEM_PROMPT}]},
        'generationConfig': {
            'temperature': temperature,
            'maxOutputTokens': max_tokens,
        },
    }
    # Auth via header (NOT URL param) so the key never lands in HTTP
    # request logs / proxies / process listings.
    url = (f'https://generativelanguage.googleapis.com/v1beta/models/'
           f'{model_cfg["model_id"]}:generateContent')
    headers = {'x-goog-api-key': api_key}
    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(5):
            try:
                resp = await client.post(url, headers=headers, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    cand = data.get('candidates', [{}])[0]
                    parts = cand.get('content', {}).get('parts', [])
                    return ''.join(p.get('text', '') for p in parts)
                if resp.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f'ERROR: HTTP {resp.status_code}: {resp.text[:200]}'
            except Exception as e:
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f'ERROR: {e}'
    return 'ERROR: max retries'


async def call_openai(prompt, image_bytes_list, model_cfg,
                       temperature, max_tokens):
    import os, httpx
    api_key = os.environ.get(model_cfg['env_key'])
    if not api_key:
        return f'ERROR: {model_cfg["env_key"]} not set'
    content = []
    for ib in image_bytes_list:
        b64 = base64.b64encode(ib).decode('ascii')
        content.append({'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{b64}',
                                       'detail': 'low'}})
    content.append({'type': 'text', 'text': prompt})
    payload = {
        'model': model_cfg['model_id'],
        'messages': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': content},
        ],
        'temperature': temperature,
        'max_completion_tokens': max_tokens,
    }
    async with httpx.AsyncClient(timeout=120) as client:
        for attempt in range(5):
            try:
                resp = await client.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={'Authorization': f'Bearer {api_key}'},
                    json=payload)
                if resp.status_code == 200:
                    return resp.json()['choices'][0]['message']['content']
                if resp.status_code in (429, 500, 502, 503):
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f'ERROR: HTTP {resp.status_code}: {resp.text[:200]}'
            except Exception as e:
                if attempt < 4:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return f'ERROR: {e}'
    return 'ERROR: max retries'


async def call_vlm(prompt, image_bytes_list, model_cfg, temperature, max_tokens):
    api = model_cfg.get('api', 'openai_compat')
    if api == 'openai_compat':
        return await call_openai_compat(prompt, image_bytes_list, model_cfg,
                                         temperature, max_tokens)
    if api == 'gemini':
        return await call_gemini(prompt, image_bytes_list, model_cfg,
                                  temperature, max_tokens)
    if api == 'openai':
        return await call_openai(prompt, image_bytes_list, model_cfg,
                                  temperature, max_tokens)
    return f'ERROR: unknown api {api}'


def parse_response(text):
    """Return {'answer': 'A'|'B'|'C'|'D', ...} or {'parse_error': True, 'raw': ...}."""
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group())
            ans = obj.get('answer', '')
            if isinstance(ans, str):
                ans = ans.strip().upper()
                if ans in ('A', 'B', 'C', 'D'):
                    obj['answer'] = ans
                    return obj
            return {**obj, 'parse_error': True, 'parse_reason': 'bad_answer_field'}
        except json.JSONDecodeError:
            pass
    # Fallback: look for a bare A/B/C/D in the response text
    m = re.search(r'\b([ABCD])\b', text)
    if m:
        return {'answer': m.group(1), 'fallback_parse': True, 'raw_text': text[:200]}
    return {'parse_error': True, 'raw': text[:200]}


# ── JSONL I/O ────────────────────────────────────────────────────────
def load_existing(path):
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r['sample_id'], r['repeat_id']))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def append_jsonl(path, rec):
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')


# ── Main run loop ────────────────────────────────────────────────────
async def run_call(sem, prompt, img_bytes_list, model_cfg,
                   sample_id, repeat_id, gt_letter, gt_face,
                   q2_strategy, input_variant, out_path, max_tokens):
    async with sem:
        raw = await call_vlm(prompt, img_bytes_list, model_cfg,
                             TEMPERATURE, max_tokens)
    parsed = parse_response(raw)
    rec = {
        'sample_id': sample_id, 'repeat_id': repeat_id,
        'q2_strategy': q2_strategy, 'input_variant': input_variant,
        'gt_letter': gt_letter, 'gt_face': gt_face,
        'raw': raw, 'parsed': parsed,
    }
    append_jsonl(out_path, rec)
    return rec


async def eval_condition_model(cond_name, queries_df, cubemap_dir,
                                model_name, model_cfg, sem,
                                cubemap_cache, frame_cache, max_per_query):
    strat, variant = parse_condition(cond_name)
    out_path = RUNS_DIR / f'{cond_name}_{model_name}_M{M_REPEATS}.jsonl'
    done = load_existing(out_path)
    expected = len(queries_df) * M_REPEATS
    logger.info(f'  {out_path.name}: {len(done)}/{expected} done')
    if len(done) >= expected:
        return

    prompt_fn = PROMPT_BUILDERS[variant]
    tasks = []
    for _, row in queries_df.iterrows():
        sid = row['sample_id']
        target = row['canonical_label']
        prompt = prompt_fn(target)

        # Load cubemap (cached)
        cm_path = str(cubemap_dir / row['image_path'])
        if cm_path in cubemap_cache:
            cm_bytes = cubemap_cache[cm_path]
        else:
            try:
                cm_bytes = load_jpeg_bytes(cm_path)
            except FileNotFoundError:
                logger.warning(f'  Cubemap missing: {cm_path}')
                continue
            cubemap_cache[cm_path] = cm_bytes

        # Compose image list per variant
        if variant == 'cubemap_only':
            img_list = [cm_bytes]
        else:  # frame_plus_cubemap
            fp = resolve_frame_path(
                row['dataset'], row['video_id'], int(row['frame_index']),
                row['participant_id'],
                bundled_frame_path=row.get('bundled_frame_path'),
                q2_strategy=row.get('q2_strategy'))
            if fp in frame_cache:
                fr_bytes = frame_cache[fp]
            else:
                try:
                    fr_bytes = load_jpeg_bytes(fp)
                except FileNotFoundError:
                    logger.warning(f'  Frame missing: {fp}')
                    continue
                # Re-encode at lower quality if very large
                frame_cache[fp] = fr_bytes
                fr_bytes = frame_cache[fp]
            img_list = [fr_bytes, cm_bytes]

        # Use larger max_tokens for paid "thinking" APIs
        max_tok = (MAX_TOKENS_API if model_cfg.get('api') in ('gemini', 'openai')
                   else MAX_TOKENS)
        for r in range(M_REPEATS):
            if (sid, r) in done:
                continue
            if max_per_query is not None and r >= max_per_query:
                continue
            tasks.append(run_call(
                sem, prompt, img_list, model_cfg,
                sid, r, row['gt_letter'], row['gt_face'],
                strat, variant, out_path, max_tok))

    if not tasks:
        return

    logger.info(f'    Queueing {len(tasks)} calls for {model_name} | {cond_name}')
    t0 = time.time()
    completed = 0
    for batch_start in range(0, len(tasks), 200):
        batch = tasks[batch_start:batch_start + 200]
        await asyncio.gather(*batch)
        completed += len(batch)
        elapsed = time.time() - t0
        rate = completed / elapsed
        eta = (len(tasks) - completed) / rate if rate > 0 else 0
        logger.info(f'    [{model_name}|{cond_name}] {completed}/{len(tasks)} '
                    f'({rate:.1f}/s, ETA {eta/60:.1f} min)')


async def main_async(args):
    logger.info('=== Q2 Eval (4 conditions × M=%d) ===', M_REPEATS)

    # Load queries per strategy. Resolution order:
    #   - Q2_DATASET_ROOT/{strategy}/queries.parquet  (HF dataset, default)
    #   - DATA_DIR/setA_extended_q2/{strategy}/queries.parquet  (legacy)
    strategy_queries = {}
    cubemap_dirs = {}
    for strat in STRATEGIES:
        path = Q2_DATASET_ROOT / strat / 'queries.parquet'
        if not path.exists():
            path = DATA_DIR / 'setA_extended_q2' / strat / 'queries.parquet'
        if not path.exists():
            logger.error(f'Missing: {path}  '
                         f'(set Q2_DATASET_ROOT or download '
                         f'nipsedtrack2026/q2-cubemap-mcq)')
            sys.exit(1)
        df = pd.read_parquet(path)
        if args.eval_set != 'full':
            from src._eval_set import filter_by_eval_set
            df = filter_by_eval_set(df, args.eval_set, id_column='q1_sample_id')
        if args.max:
            df = df.head(args.max)
        strategy_queries[strat] = df.reset_index(drop=True)
        cubemap_dirs[strat] = path.parent
        logger.info(f'  {strat}: {len(df)} queries (eval_set={args.eval_set})')

    # Resolve conditions and models
    conditions = [c for c in args.conditions if c in CONDITIONS]
    if not conditions:
        logger.error(f'No valid conditions in {args.conditions}; '
                     f'options: {CONDITIONS}')
        sys.exit(1)
    logger.info(f'  Conditions: {conditions}')

    models = {m: MODELS[m] for m in args.models if m in MODELS}
    if not models:
        logger.error(f'No valid models in {args.models}; options: {list(MODELS)}')
        sys.exit(1)
    logger.info(f'  Models: {list(models)}')

    if args.dry_run:
        for c in conditions:
            strat, _ = parse_condition(c)
            n = len(strategy_queries[strat])
            for m in models:
                p = RUNS_DIR / f'{c}_{m}_M{M_REPEATS}.jsonl'
                done = load_existing(p)
                expected = n * M_REPEATS
                logger.info(f'    {c} {m}: {len(done)}/{expected}')
        return

    # Server health checks (only for local openai_compat servers)
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        for mname, mcfg in models.items():
            if mcfg.get('api') != 'openai_compat':
                logger.info(f'  {mname} (api={mcfg["api"]}) — skipping health check')
                continue
            url = mcfg['base_url'].rsplit('/v1/', 1)[0] + '/v1/models'
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    logger.warning(f'  {mname} server unhealthy: {r.status_code}')
                else:
                    served = [x['id'] for x in r.json().get('data', [])]
                    logger.info(f'  {mname} OK ({served})')
            except Exception as e:
                logger.warning(f'  {mname} server check failed: {e}')

    cubemap_cache = {}
    frame_cache = {}
    sems = {m: asyncio.Semaphore(
        API_CONCURRENCY if mcfg.get('api') in ('gemini', 'openai')
        else CONCURRENCY_PER_MODEL)
        for m, mcfg in models.items()}

    async def run_model_queue(m_name, m_cfg, sem):
        for c_name in conditions:
            strat, _ = parse_condition(c_name)
            df = strategy_queries[strat]
            cmd = cubemap_dirs[strat]
            logger.info(f'\n[{m_name}] {c_name}')
            await eval_condition_model(c_name, df, cmd, m_name, m_cfg, sem,
                                        cubemap_cache, frame_cache,
                                        args.m_per_query)
        logger.info(f'\n[{m_name}] DONE')

    await asyncio.gather(*[
        run_model_queue(m_name, m_cfg, sems[m_name])
        for m_name, m_cfg in models.items()
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conditions', nargs='+', default=CONDITIONS)
    parser.add_argument('--models', nargs='+',
                        default=list(MODELS.keys()))
    parser.add_argument('--max', type=int, default=None,
                        help='Limit queries (smoke test).')
    parser.add_argument('--m-per-query', type=int, default=None,
                        help='Cap repeats below M_REPEATS (smoke test).')
    parser.add_argument('--dry-run', action='store_true')
    from src._eval_set import add_eval_set_arg
    add_eval_set_arg(parser)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
