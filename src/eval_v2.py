"""
Evaluation pipeline for the v2 sampling-method ablation (2026-04-25).

Runs M=25 samples per query × 4 conditions × 3 models × 2 sets on the
4bin_height bin scheme (9 joint bins).

Conditions:
  baseline_sighted          : single egocentric image, no panorama
  baseline_blind_strict     : no image at all (label-only)
  baseline_sighted_cot_E    : egocentric image + CoT-E reasoning template
  cube4_cautious_generic    : panorama (cube4 layout) + cautious + generic outpaint

Models (vLLM endpoints):
  qwen3.5-9b   : http://127.0.0.1:8001/v1/chat/completions
  gemma-4-31b  : http://127.0.0.1:9002/v1/chat/completions
  qwen3-vl-30b : http://127.0.0.1:18002/v1/chat/completions

Output (one JSONL per (set, condition, model)):
  meta/testset/exp/sampling_ablation_001/runs/{set}_{condition}_{model}_M25.jsonl
  Each line: {sample_id, repeat_id, label, gt_yaw, gt_height,
              raw, parsed}

Run:
  conda run -n slam python3 -m src.eval_v2 --sets A B --conditions sighted blind cot-e \
      --models qwen3-vl-30b gemma-4-31b
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
# All paths are env-driven so a fresh clone + `huggingface-cli download
# nipsedtrack2026/q1-bin-prediction --local-dir data/q1` followed by
# `python -m src.eval_v2` works out of the box from the eval-code root.
TESTSET = Path(os.environ.get(
    "TESTSET_ROOT",
    str(Path(__file__).resolve().parents[1]),  # default: parent of src/
))
Q1_DATASET_ROOT = Path(os.environ.get(
    "Q1_DATASET_ROOT",
    str(TESTSET / "data" / "q1"),  # default: ./data/q1
))
DATA_DIR = TESTSET / "data"
RUNS_DIR = Path(os.environ.get(
    "RUNS_DIR",
    str(TESTSET / "runs"),  # default: ./runs
))
RUNS_DIR.mkdir(parents=True, exist_ok=True)

# Legacy upstream-frame roots — only used when bundled_frame_path is
# missing from the queries parquet (i.e. you're not using the HF dataset).
EK_ROOT = Path(os.environ.get(
    "EK_FRAMES_ROOT",
    "/path/to/epic-kitchens",
))
HD_ROOT = Path(os.environ.get(
    "HD_EPIC_FRAMES_ROOT",
    "/path/to/hd-epic/Participants",
))

# ── Model endpoints ───────────────────────────────────────────────────
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

# ── Eval config ───────────────────────────────────────────────────────
M_REPEATS = 25
TEMPERATURE = 1.0
MAX_TOKENS = 512
MAX_TOKENS_COT = 1024
CONCURRENCY_PER_MODEL = 8

# ── Prompts (4-bin yaw × 3-bin height = 9 joint bins) ────────────────
SYSTEM_PROMPT = "You are a careful spatial reasoner."

YAW_DESC = (
    "There are 4 direction bins (90° each, clockwise from camera forward):\n"
    "  Bin 0 = Front (12 o'clock — EXCLUDED, target must be out of view)\n"
    "  Bin 1 = Right (3 o'clock)\n"
    "  Bin 2 = Back (6 o'clock)\n"
    "  Bin 3 = Left (9 o'clock)"
)

VERT_DESC = (
    "And 3 height levels relative to the kitchen benchtop:\n"
    "  UP    = above benchtop (wall cabinets, shelves, range hood, wall-mounted items)\n"
    "  LEVEL = on benchtop / work zone (countertop appliances, utensils, sink)\n"
    "  DOWN  = below benchtop (oven, lower cabinets, dishwasher, trash bin, floor)"
)


def build_sighted_prompt(label: str) -> str:
    return f"""\
You are given ONE egocentric RGB image (first-person view).
The target object "{label}" is OUT OF VIEW (not visible in the image).
Predict its most likely direction based on visible layout cues \
(walls, counters, appliances, doorways, free space) and spatial commonsense.

{YAW_DESC}

{VERT_DESC}

Output a single JSON object:
  {{"justification": "<1-2 sentences of spatial reasoning>",
    "yaw_bin_id": <integer 1-3>,
    "pitch": "<UP|LEVEL|DOWN>"}}

Example:
  {{"justification": "The sink is typically behind and to the left in this layout.",
    "yaw_bin_id": 2, "pitch": "LEVEL"}}
"""


def build_cot_e_prompt(label: str) -> str:
    return f"""\
You are given ONE egocentric RGB image (first-person view).
The target object "{label}" is OUT OF VIEW (not visible in the image).
Predict its most likely direction based on visible layout cues \
(walls, counters, appliances, doorways, free space) and spatial commonsense.

{YAW_DESC}

{VERT_DESC}

Think step by step:
1. Describe what you see in the image: key surfaces, appliances, and layout.
2. Imagine what is likely to the LEFT of the image (just outside the left edge).
3. Imagine what is likely to the RIGHT of the image (just outside the right edge).
4. Imagine what is likely BEHIND you (opposite to where you are looking).
5. Given your mental picture of the full kitchen, where is "{label}" most \
likely located? Predict its direction and height.

Output a single JSON object:
  {{"visible_scene": "<what you see>",
    "left_of_frame": "<what is likely to the left>",
    "right_of_frame": "<what is likely to the right>",
    "behind_camera": "<what is likely behind>",
    "prediction_reasoning": "<why {label} is in this direction>",
    "yaw_bin_id": <integer 1-3>,
    "pitch": "<UP|LEVEL|DOWN>"}}
"""


CONDITIONS = {
    'sighted':  {'prompt_fn': build_sighted_prompt, 'send_image': True,  'max_tokens': MAX_TOKENS},
    'blind':    {'prompt_fn': build_sighted_prompt, 'send_image': False, 'max_tokens': MAX_TOKENS},
    'cot-e':    {'prompt_fn': build_cot_e_prompt,   'send_image': True,  'max_tokens': MAX_TOKENS_COT},
    # cube4-cautious-generic excluded for now (panorama assets missing
    # for many of our v2 frames; needs separate panorama-generation pass)
}


# ── Frame I/O ────────────────────────────────────────────────────────
def resolve_frame_path(dataset: str, video_id: str, frame_index: int,
                       bundled_frame_path: str | None = None) -> str:
    """Resolve a frame to an absolute path.

    Preference: `bundled_frame_path` (relative path shipped by the HF
    q1-bin-prediction dataset under `frames/`). Falls back to the legacy
    upstream-style EK/HD layout when no bundled path is available (e.g.
    when running against an out-of-tree custom queries parquet).
    """
    if bundled_frame_path:
        return str(Q1_DATASET_ROOT / bundled_frame_path)
    if dataset == 'epic_kitchens':
        pid = video_id.split('_')[0]
        return str(EK_ROOT / pid / video_id / 'frames'
                   / f'frame_{frame_index:010d}.jpg')
    else:
        pid = video_id.split('-')[0]
        return str(HD_ROOT / pid / video_id / 'images'
                   / f'frame_{frame_index:06d}.jpg')


def img_to_jpeg_bytes(img_rgb, quality=90):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


# ── VLM call ─────────────────────────────────────────────────────────
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
    import httpx
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
    import httpx
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
    if api == 'gemini':
        return await call_gemini(prompt, image_bytes_list, model_cfg,
                                  temperature, max_tokens)
    if api == 'openai':
        return await call_openai(prompt, image_bytes_list, model_cfg,
                                  temperature, max_tokens)
    return await call_openai_compat(prompt, image_bytes_list, model_cfg,
                                     temperature, max_tokens)


def parse_response(text):
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {'raw': text, 'parse_error': True}


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
                   sample_id, repeat_id, label, gt_yaw, gt_height,
                   out_path, max_tokens):
    async with sem:
        raw = await call_vlm(prompt, img_bytes_list, model_cfg,
                             TEMPERATURE, max_tokens)
    parsed = parse_response(raw)
    rec = {'sample_id': sample_id, 'repeat_id': repeat_id,
           'label': label, 'gt_yaw': int(gt_yaw),
           'gt_height': int(gt_height), 'raw': raw, 'parsed': parsed}
    append_jsonl(out_path, rec)
    return rec


async def eval_set_condition_model(set_name, set_df, condition_name,
                                   condition_cfg, model_name, model_cfg,
                                   frame_cache, sem):
    out_path = RUNS_DIR / f'{set_name}_{condition_name}_{model_name}_M{M_REPEATS}.jsonl'
    done = load_existing(out_path)
    expected = len(set_df) * M_REPEATS
    logger.info(f'  {out_path.name}: {len(done)}/{expected} done')
    if len(done) >= expected:
        return

    tasks = []
    for _, row in set_df.iterrows():
        label = row['canonical_label']
        prompt = condition_cfg['prompt_fn'](label)
        # Image bytes
        if condition_cfg['send_image']:
            fp = resolve_frame_path(
                row['dataset'], row['video_id'], int(row['frame_index']),
                bundled_frame_path=row.get('bundled_frame_path'))
            if fp in frame_cache:
                img_list = [frame_cache[fp]]
            else:
                if not Path(fp).exists():
                    logger.warning(f'  Frame missing: {fp}')
                    continue
                img_bgr = cv2.imread(fp)
                if img_bgr is None:
                    logger.warning(f'  Frame unreadable: {fp}')
                    continue
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                jpeg = img_to_jpeg_bytes(img_rgb)
                frame_cache[fp] = jpeg
                img_list = [jpeg]
        else:
            img_list = []

        for r in range(M_REPEATS):
            if (row['sample_id'], r) in done:
                continue
            tasks.append(run_call(
                sem, prompt, img_list, model_cfg,
                row['sample_id'], r, label,
                row['yaw_bin_4'], row['height_bin'],
                out_path, condition_cfg['max_tokens']))

    if not tasks:
        return

    logger.info(f'    Queueing {len(tasks)} calls')
    t0 = time.time()
    completed = 0
    for batch_start in range(0, len(tasks), 200):
        batch = tasks[batch_start:batch_start + 200]
        await asyncio.gather(*batch)
        completed += len(batch)
        elapsed = time.time() - t0
        rate = completed / elapsed
        eta = (len(tasks) - completed) / rate if rate > 0 else 0
        logger.info(f'    {completed}/{len(tasks)} done ({rate:.1f}/s, '
                    f'ETA {eta/60:.1f} min)')


async def main_async(args):
    global M_REPEATS
    if args.m_repeats is not None:
        M_REPEATS = args.m_repeats
    logger.info(f'=== Eval v2 (sampling ablation) — M={M_REPEATS} ===')

    # Load sets. Resolution order for each `--sets` value:
    #   - 'full' or 'q1' → Q1_DATASET_ROOT/queries.parquet (HF dataset)
    #   - explicit path  → use as-is
    #   - bare letter    → DATA_DIR/set{X}_300_filtered.parquet (legacy)
    sets = {}
    for s in args.sets:
        if s in ('full', 'q1'):
            path = Q1_DATASET_ROOT / 'queries.parquet'
        elif Path(s).is_absolute() or '/' in s:
            path = Path(s)
        else:
            path = DATA_DIR / f'set{s}_300_filtered.parquet'
        if not path.exists():
            logger.error(f'Missing: {path}  '
                         f'(set Q1_DATASET_ROOT or pass an explicit path)')
            sys.exit(1)
        sets[s] = pd.read_parquet(path)
        if args.eval_set != 'full':
            from src._eval_set import filter_by_eval_set
            sets[s] = filter_by_eval_set(sets[s], args.eval_set)
        logger.info(f'  Set {s} ({path.name}): {len(sets[s])} queries '
                    f'(eval_set={args.eval_set})')

    # Resolve conditions
    conditions = {c: CONDITIONS[c] for c in args.conditions if c in CONDITIONS}
    if not conditions:
        logger.error(f'No valid conditions in {args.conditions}')
        sys.exit(1)
    logger.info(f'  Conditions: {list(conditions.keys())}')

    # Resolve models
    models = {m: MODELS[m] for m in args.models if m in MODELS}
    if not models:
        logger.error(f'No valid models in {args.models}')
        sys.exit(1)
    logger.info(f'  Models: {list(models.keys())}')

    if args.dry_run:
        total = (len(args.sets) * len(conditions) * len(models)
                 * 300 * M_REPEATS)
        logger.info(f'  Total calls (dry-run): {total}')
        for s in args.sets:
            for c in conditions:
                for m in models:
                    p = RUNS_DIR / f'{s}_{c}_{m}_M{M_REPEATS}.jsonl'
                    done = load_existing(p)
                    expected = len(sets[s]) * M_REPEATS
                    logger.info(f'    set{s} {c} {m}: {len(done)}/{expected}')
        return

    # Server health checks (vLLM-style endpoints only; APIs are checked
    # at first request).
    import httpx
    async with httpx.AsyncClient(timeout=10) as client:
        for mname, mcfg in models.items():
            if mcfg.get('api', 'openai_compat') != 'openai_compat':
                # Gemini/OpenAI API: no /v1/models pre-check; warn if env key missing.
                if mcfg.get('env_key') and not os.environ.get(mcfg['env_key']):
                    logger.warning(f'  {mname}: env var {mcfg["env_key"]} '
                                   f'not set — calls will fail')
                else:
                    logger.info(f'  {mname} (api={mcfg["api"]}): env OK')
                continue
            url = mcfg['base_url'].rsplit('/v1/', 1)[0] + '/v1/models'
            try:
                r = await client.get(url)
                if r.status_code != 200:
                    logger.warning(f'  {mname} server unhealthy: HTTP {r.status_code}')
                else:
                    served = [m['id'] for m in r.json().get('data', [])]
                    if mcfg['model_id'] not in served:
                        logger.warning(
                            f'  {mname}: expected {mcfg["model_id"]}, '
                            f'server has {served}')
                    else:
                        logger.info(f'  {mname} server OK ({served})')
            except Exception as e:
                logger.warning(f'  {mname} server check failed: {e}')

    # Frame cache
    frame_cache = {}

    # Per-model semaphore
    sems = {m: asyncio.Semaphore(CONCURRENCY_PER_MODEL) for m in models}

    # Per-model independent queues: each model walks through all
    # (set, condition) cells on its own. Idle models don't wait for
    # slow cells. resume-from-existing-jsonl skips any cell already done.
    async def run_model_queue(m_name, m_cfg, sem):
        for s_name, s_df in sets.items():
            for c_name, c_cfg in conditions.items():
                logger.info(f'\n[{m_name}] Set {s_name} | {c_name}')
                await eval_set_condition_model(
                    s_name, s_df, c_name, c_cfg, m_name, m_cfg,
                    frame_cache, sem)
        logger.info(f'\n[{m_name}] DONE — all cells complete for this model')

    await asyncio.gather(*[
        run_model_queue(m_name, m_cfg, sems[m_name])
        for m_name, m_cfg in models.items()
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sets', nargs='+', default=['full'],
                        help="One or more set ids. 'full' / 'q1' load "
                             "Q1_DATASET_ROOT/queries.parquet (HF dataset, "
                             "default). 'A' / 'B' load the legacy "
                             "set{X}_300_filtered.parquet. An explicit "
                             "path is also accepted.")
    parser.add_argument('--conditions', nargs='+',
                        default=['sighted', 'blind', 'cot-e'])
    parser.add_argument('--models', nargs='+',
                        default=['qwen3.5-9b'])
    parser.add_argument('--m-repeats', '--m_repeats', dest='m_repeats',
                        type=int, default=None,
                        help='Override M (samples per query). '
                             f'Default {25}.')
    parser.add_argument('--dry-run', action='store_true')
    from src._eval_set import add_eval_set_arg
    add_eval_set_arg(parser)
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == '__main__':
    main()
