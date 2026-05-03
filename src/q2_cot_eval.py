"""
Q2 CoT evaluation: CoT-B through CoT-H on Q2 multiple-choice cubemap.

Uses A_random_diff_face strategy + frame_plus_cubemap variant for all CoTs.
Distributes work across multiple model endpoints for speed.

CoT types:
  B  — Anchor reasoning (list anchors + spatial relationships)
  B2 — Merged anchor reasoning (combined identification + relationship)
  C  — Easy-to-hard elimination (height → eliminate → yaw)
  D  — Height-first (height → yaw, no elimination)
  E  — Imagine 360° surroundings
  F  — Zero-shot "Let's think step by step" (T=0, M=1)
  G  — Self-consistency (same as F but T=1.0, M=5, majority vote)
  H  — Tree-of-Thought (2-stage: plan T=0.7, then 3 answers T=1.0)

Run:
  conda run -n slam python3 -m src.q2_cot_eval
  conda run -n slam python3 -m src.q2_cot_eval --cots F G --models qwen3.5-9b
  conda run -n slam python3 -m src.q2_cot_eval --max 10  # smoke test
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")

# Reuse infrastructure from q2_eval
sys.path.insert(0, str(Path(__file__).parent))
from q2_eval import (
    TESTSET, DATA_DIR, RUNS_DIR, SYSTEM_PROMPT,
    LAYOUT_DESC, OPTIONS_DESC,
    call_vlm, parse_response, load_existing, append_jsonl,
    resolve_frame_path, load_jpeg_bytes,
)

# ── Model endpoints (load-balanced across cluster + vast.ai) ──────────

MODELS = {
    'qwen3.5-9b': {
        'api': 'openai_compat',
        'base_url': 'http://127.0.0.1:8001/v1/chat/completions',
        'model_id': 'Qwen/Qwen3.5-9B',
        'extra_body': {'chat_template_kwargs': {'enable_thinking': False}},
    },
    'gemma-4-31b': {
        'api': 'openai_compat',
        'base_url': 'http://127.0.0.1:9006/v1/chat/completions',
        'model_id': 'google/gemma-4-31b-it',
        'extra_body': {},
    },
    'qwen3-vl-30b': {
        # Use vast.ai endpoint (9005) — cluster (9003) as fallback
        'api': 'openai_compat',
        'base_url': 'http://127.0.0.1:9005/v1/chat/completions',
        'model_id': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'extra_body': {},
    },
}

# Secondary endpoints for load balancing
SECONDARY_ENDPOINTS = {
    'qwen3-vl-30b': {
        'api': 'openai_compat',
        'base_url': 'http://127.0.0.1:9003/v1/chat/completions',
        'model_id': 'Qwen/Qwen3-VL-30B-A3B-Instruct',
        'extra_body': {},
    },
}

CONCURRENCY = 8
STRATEGY = 'A_random_diff_face'

# ── CoT Prompt Builders ───────────────────────────────────────────────

def _base_opening(target):
    return (
        f'You are given TWO images: (1) the original egocentric RGB image '
        f'(first-person view), and (2) the same image projected as a 2x3 '
        f'cubemap from the camera\'s viewpoint.\n'
        f'The target object "{target}" is OUT OF VIEW (not visible in either image).\n\n'
        f'{LAYOUT_DESC}\n\n'
        f'{OPTIONS_DESC}'
    )


def build_cot_B(target):
    return f"""{_base_opening(target)}

Step 1: List the 3-5 most prominent objects/surfaces visible in the cubemap, with their approximate position (which tile: FRONT/LEFT/BEHIND/RIGHT/UP/DOWN, and where within the tile).

Step 2: For each visible object you listed, state its typical spatial relationship to "{target}" in a kitchen (e.g., "sinks are often near stoves", "dishwashers are usually under the counter near the sink").

Step 3: Based on the anchor positions and relationships, predict which marker (A, B, C, or D) is most likely at the location of "{target}".

Output a single JSON object:
  {{"visible_anchors": "<list of visible objects and positions>",
    "anchor_reasoning": "<spatial relationships to target>",
    "answer": "<A|B|C|D>"}}
"""


def build_cot_B2(target):
    return f"""{_base_opening(target)}

Step 1: Look at the cubemap carefully. For each prominent visible object, describe its position AND reason about its spatial relationship to "{target}" in typical kitchens.

Step 2: Given all the relationships you identified, look at the marker positions. Which marker is most consistent with where "{target}" should be?

Output a single JSON object:
  {{"anchor_reasoning": "<objects, positions, and relationships to target>",
    "prediction_reasoning": "<why the chosen marker is best>",
    "answer": "<A|B|C|D>"}}
"""


def build_cot_C(target):
    return f"""{_base_opening(target)}

Step 1: Look at the cubemap. Identify which height zone "{target}" typically belongs to:
  UP = above benchtop (wall cabinets, shelves, range hood)
  LEVEL = on benchtop (countertop appliances, utensils, sink)
  DOWN = below benchtop (oven, lower cabinets, dishwasher, trash bin, floor)
Which height zone is most likely for "{target}"?

Step 2: Looking at the markers, eliminate any that are clearly in the WRONG height zone for "{target}". Which markers can you rule out?

Step 3: Among the remaining markers, which is in the most plausible direction for "{target}" given the visible kitchen layout?

Output a single JSON object:
  {{"height_reasoning": "<height zone analysis>",
    "eliminated": "<which markers eliminated and why>",
    "answer": "<A|B|C|D>"}}
"""


def build_cot_D(target):
    return f"""{_base_opening(target)}

Step 1: Look at the cubemap. Predict the height level of "{target}":
  UP = above benchtop | LEVEL = on benchtop | DOWN = below benchtop

Step 2: Given your height prediction, look at the cubemap again. Which marker (A, B, C, or D) is at the right height AND in the most plausible direction for "{target}"?

Output a single JSON object:
  {{"height_reasoning": "<height prediction and why>",
    "direction_reasoning": "<direction prediction given height>",
    "answer": "<A|B|C|D>"}}
"""


def build_cot_E(target):
    return f"""{_base_opening(target)}

Before predicting, build a mental model of the full kitchen:

Step 1: Describe what's visible in the FRONT tile of the cubemap (the camera's forward view).
Step 2: What would likely be to the LEFT (outside the left edge)?
Step 3: What would likely be to the RIGHT (outside the right edge)?
Step 4: What's likely BEHIND the camera?
Step 5: Given your 360° mental picture, which marker (A, B, C, or D) is at the most likely location of "{target}"?

Output a single JSON object:
  {{"visible_scene": "<what's in FRONT>",
    "left_of_frame": "<what's LEFT>",
    "right_of_frame": "<what's RIGHT>",
    "behind_camera": "<what's BEHIND>",
    "answer": "<A|B|C|D>"}}
"""


def build_cot_F(target):
    return f"""{_base_opening(target)}

Let's think step by step about where "{target}" would be located.

Output a single JSON object:
  {{"reasoning": "<step-by-step reasoning>",
    "answer": "<A|B|C|D>"}}
"""


# CoT-G uses same prompt as F but different M and T (handled in runner)
build_cot_G = build_cot_F


def build_cot_H_plan(target):
    """Stage 1: plan generation."""
    return f"""{_base_opening(target)}

Before answering, draft a reasoning plan. Consider:
- What visible landmarks or spatial cues in the cubemap are relevant to locating "{target}"?
- What is your reasoning strategy? (e.g., use landmark co-location, room layout, elimination of implausible positions)

Output a single JSON object:
  {{"plan": "<your reasoning plan in 2-3 sentences>"}}
"""


def build_cot_H_answer(target, plan):
    """Stage 2: answer generation following plan."""
    return f"""{_base_opening(target)}

Follow this reasoning plan: {plan}

Based on this plan, predict which marker (A, B, C, or D) is at the target's location.

Output a single JSON object:
  {{"reasoning": "<reasoning following the plan>",
    "confidence": "<high|medium|low>",
    "answer": "<A|B|C|D>"}}
"""


# ── CoT configs ───────────────────────────────────────────────────────

COT_CONFIGS = {
    'B':  {'builder': build_cot_B,  'M': 1, 'T': 0.0, 'max_tokens': 1024, 'stages': 1},
    'B2': {'builder': build_cot_B2, 'M': 1, 'T': 0.0, 'max_tokens': 1024, 'stages': 1},
    'C':  {'builder': build_cot_C,  'M': 1, 'T': 0.0, 'max_tokens': 1024, 'stages': 1},
    'D':  {'builder': build_cot_D,  'M': 1, 'T': 0.0, 'max_tokens': 1024, 'stages': 1},
    'E':  {'builder': build_cot_E,  'M': 1, 'T': 0.0, 'max_tokens': 1024, 'stages': 1},
    'F':  {'builder': build_cot_F,  'M': 1, 'T': 0.0, 'max_tokens': 1024, 'stages': 1},
    'G':  {'builder': build_cot_G,  'M': 5, 'T': 1.0, 'max_tokens': 1024, 'stages': 1},
    'H':  {'builder': None,         'M': 1, 'T': None, 'max_tokens': 1024, 'stages': 2},
}


# ── Run helpers ───────────────────────────────────────────────────────

async def run_single_stage(sem, prompt, img_list, model_cfg, temperature, max_tokens,
                           sample_id, repeat_id, gt_letter, gt_face, cot_name, out_path):
    async with sem:
        raw = await call_vlm(prompt, img_list, model_cfg, temperature, max_tokens)
    parsed = parse_response(raw)
    rec = {
        'sample_id': sample_id, 'repeat_id': repeat_id,
        'cot': cot_name, 'gt_letter': gt_letter, 'gt_face': gt_face,
        'raw': raw, 'parsed': parsed,
    }
    append_jsonl(out_path, rec)
    return rec


async def run_cot_H(sem, img_list, model_cfg, target,
                    sample_id, gt_letter, gt_face, out_path):
    """Two-stage Tree-of-Thought: plan then 3 answers.

    Acquires sem ONCE for the entire 4-call sequence to avoid deadlock
    when many H tasks compete for semaphore slots.
    """
    async with sem:
        # Stage 1: plan (T=0.7)
        plan_prompt = build_cot_H_plan(target)
        plan_raw = await call_vlm(plan_prompt, img_list, model_cfg, 0.7, 512)
        # Extract plan text
        plan_text = ""
        try:
            m = re.search(r'\{.*\}', plan_raw, re.DOTALL)
            if m:
                obj = json.loads(m.group())
                plan_text = obj.get('plan', plan_raw[:200])
        except:
            plan_text = plan_raw[:200]

        # Stage 2: 3 candidate answers (T=1.0)
        ans_prompt = build_cot_H_answer(target, plan_text)
        candidates = []
        for i in range(3):
            ans_raw = await call_vlm(ans_prompt, img_list, model_cfg, 1.0, 1024)
        parsed = parse_response(ans_raw)
        confidence = parsed.get('confidence', 'low') if not parsed.get('parse_error') else 'none'
        candidates.append({'raw': ans_raw, 'parsed': parsed, 'confidence': confidence})

    # Select: highest confidence, then majority vote if tied
    conf_order = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
    candidates.sort(key=lambda c: conf_order.get(c['confidence'], 0), reverse=True)
    best = candidates[0]

    rec = {
        'sample_id': sample_id, 'repeat_id': 0,
        'cot': 'H', 'gt_letter': gt_letter, 'gt_face': gt_face,
        'plan': plan_text,
        'candidates': [{'answer': c['parsed'].get('answer'), 'confidence': c['confidence']}
                       for c in candidates],
        'raw': best['raw'], 'parsed': best['parsed'],
    }
    append_jsonl(out_path, rec)
    return rec


# ── Main ──────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cots', nargs='*', default=list(COT_CONFIGS.keys()),
                        help='CoT types to run (default: all)')
    parser.add_argument('--models', nargs='*', default=list(MODELS.keys()))
    parser.add_argument('--max', type=int, default=None, help='Max queries (smoke test)')
    parser.add_argument('--override-url', default=None,
                        help='Override base_url for all models (e.g., http://127.0.0.1:9003/v1/chat/completions)')
    from src._eval_set import add_eval_set_arg, filter_by_eval_set
    add_eval_set_arg(parser)
    args = parser.parse_args()

    # Load queries
    strat_dir = DATA_DIR / f'setA_extended_q2' / STRATEGY
    queries = pd.read_parquet(strat_dir / 'queries.parquet')
    if args.eval_set != 'full':
        queries = filter_by_eval_set(queries, args.eval_set, id_column='q1_sample_id')
    if args.max:
        queries = queries.head(args.max)
    logger.info(f'Loaded {len(queries)} queries from {STRATEGY} (eval_set={args.eval_set})')

    # Pre-load cubemaps and frames
    logger.info('Pre-loading images...')
    cubemap_cache = {}
    frame_cache = {}
    for _, row in queries.iterrows():
        cm_path = str(strat_dir / row['image_path'])
        if cm_path not in cubemap_cache:
            try:
                cubemap_cache[cm_path] = load_jpeg_bytes(cm_path)
            except FileNotFoundError:
                pass
        fp = resolve_frame_path(row['dataset'], row['video_id'],
                                int(row['frame_index']), row['participant_id'])
        if fp not in frame_cache:
            try:
                frame_cache[fp] = load_jpeg_bytes(fp)
            except FileNotFoundError:
                pass
    logger.info(f'  {len(cubemap_cache)} cubemaps, {len(frame_cache)} frames')

    t0_global = time.time()

    for cot_name in args.cots:
        cfg = COT_CONFIGS[cot_name]
        M = cfg['M']
        T = cfg['T']
        max_tokens = cfg['max_tokens']

        for model_name in args.models:
            if model_name not in MODELS:
                logger.warning(f'Unknown model: {model_name}')
                continue
            model_cfg = dict(MODELS[model_name])
            if args.override_url:
                model_cfg['base_url'] = args.override_url
            cond_label = f'A_fpc_cot_{cot_name}'
            out_path = RUNS_DIR / f'{cond_label}_{model_name}_M{M}.jsonl'
            done = load_existing(out_path)
            expected = len(queries) * M
            logger.info(f'{cond_label}/{model_name}: {len(done)}/{expected} done')

            if len(done) >= expected:
                continue

            sem = asyncio.Semaphore(CONCURRENCY)
            tasks = []
            t0 = time.time()

            for _, row in queries.iterrows():
                sid = row['sample_id']
                target = row['canonical_label']
                gt_letter = row['gt_letter']
                gt_face = row['gt_face']

                cm_path = str(strat_dir / row['image_path'])
                fp = resolve_frame_path(row['dataset'], row['video_id'],
                                        int(row['frame_index']), row['participant_id'])
                if cm_path not in cubemap_cache or fp not in frame_cache:
                    continue
                img_list = [frame_cache[fp], cubemap_cache[cm_path]]

                if cot_name == 'H':
                    # 2-stage: handled specially
                    if (sid, 0) in done:
                        continue
                    tasks.append(run_cot_H(
                        sem, img_list, model_cfg, target,
                        sid, gt_letter, gt_face, out_path))
                else:
                    prompt = cfg['builder'](target)
                    for r in range(M):
                        if (sid, r) in done:
                            continue
                        tasks.append(run_single_stage(
                            sem, prompt, img_list, model_cfg, T, max_tokens,
                            sid, r, gt_letter, gt_face, cot_name, out_path))

            if not tasks:
                continue

            logger.info(f'  Launching {len(tasks)} calls...')
            completed = 0
            for i in range(0, len(tasks), 200):
                batch = tasks[i:i+200]
                await asyncio.gather(*batch)
                completed += len(batch)
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - completed) / rate if rate > 0 else 0
                logger.info(f'  [{model_name}|cot_{cot_name}] {completed}/{len(tasks)} '
                            f'({rate:.1f}/s, ETA {eta/60:.1f}min)')

    elapsed = time.time() - t0_global
    logger.info(f'\nAll done in {elapsed/60:.1f}min')


if __name__ == '__main__':
    asyncio.run(main())
