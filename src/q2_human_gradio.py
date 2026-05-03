"""
Gradio app for the human Q2 trial.

Pick 100 queries from setA_extended_q2/A_random_diff_face. Show the user:
  - Original RGB frame (left)
  - 2x3 cubemap with markers (right)
  - Prompt + target object name
  - 4 buttons (A/B/C/D), Skip, Restart

Track running accuracy + per-letter pick distribution. The user can
see "correct/wrong + GT letter" feedback after each pick (or hide
feedback to do the trial blind first — toggle in UI).

Run:
  conda run -n slam python3 -m src.q2_human_gradio
  conda run -n slam python3 -m src.q2_human_gradio --port 17861 --n 100
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image

TESTSET = Path("/path/to/this/repo")
DATA_DIR = TESTSET / "data"
Q2_DIR = DATA_DIR / "setA_extended_q2" / "A_random_diff_face"
TRIAL_LOG_DIR = TESTSET / "exp" / "q2_eval_001" / "human_trial"
TRIAL_LOG_DIR.mkdir(parents=True, exist_ok=True)

EK_ROOT = Path("/path/to/epic-kitchens")
HD_ROOT = Path(
    "/path/to/hd-epic/Participants"
)


def resolve_frame_path(dataset, video_id, frame_index, participant_id):
    if dataset == 'epic_kitchens':
        return EK_ROOT / participant_id / video_id / 'frames' / f'frame_{frame_index:010d}.jpg'
    return HD_ROOT / participant_id / video_id / 'images' / f'frame_{frame_index:06d}.jpg'


def make_app(n_samples: int, seed: int):
    queries_df = pd.read_parquet(Q2_DIR / "queries.parquet")
    rng = np.random.RandomState(seed)
    chosen_idx = sorted(rng.choice(len(queries_df), size=min(n_samples, len(queries_df)),
                                    replace=False).tolist())
    queries = queries_df.iloc[chosen_idx].reset_index(drop=True)
    print(f"Picked {len(queries)} queries (seed={seed})")

    # State holders (lists so they're mutable from event handlers)
    state = {
        'idx': 0,
        'records': [],   # one dict per answered question
        'show_feedback': True,
    }

    def load_query(i):
        if i >= len(queries):
            return None
        row = queries.iloc[i]
        cubemap_path = Q2_DIR / row['image_path']
        frame_path = resolve_frame_path(
            row['dataset'], row['video_id'], int(row['frame_index']),
            row['participant_id'])
        cubemap = Image.open(cubemap_path).convert('RGB')
        try:
            frame = Image.open(frame_path).convert('RGB')
        except FileNotFoundError:
            frame = Image.new('RGB', (320, 240), color=(80, 80, 80))
        return {
            'idx': i,
            'sample_id': row['sample_id'],
            'target': row['canonical_label'],
            'gt_letter': row['gt_letter'],
            'gt_face': row['gt_face'],
            'frame': frame,
            'cubemap': cubemap,
            'prompt': row['prompt'],
        }

    def render_status(idx):
        n_done = len(state['records'])
        n_total = len(queries)
        if n_done == 0:
            acc_text = '— no answers yet —'
        else:
            n_correct = sum(1 for r in state['records'] if r['correct'])
            acc = n_correct / n_done
            picks = [r['picked'] for r in state['records']]
            pick_dist = ' '.join(f'{L}:{picks.count(L)}' for L in 'ABCD')
            acc_text = (f'**{n_correct}/{n_done} = {acc:.1%}** correct • '
                         f'picks: {pick_dist}')
        return (f'### Query {idx + 1} / {n_total}\n\n'
                 f'{acc_text}')

    def get_initial_view():
        q = load_query(state['idx'])
        if q is None:
            return ('Done!', None, None, '', '', gr.update(visible=False))
        status = render_status(state['idx'])
        question = (f'**Target:** *{q["target"]}*\n\n'
                     f'(Sample {q["sample_id"]})\n\n'
                     f'Where is the target? Pick A/B/C/D, or Skip.')
        feedback = '_(no answer yet)_'
        return (status, q['frame'], q['cubemap'], question,
                feedback, gr.update(visible=False))

    def pick_letter(letter):
        if state['idx'] >= len(queries):
            return ('Done!', None, None, '', '_(trial complete)_',
                    gr.update(visible=False))
        q = load_query(state['idx'])
        correct = (letter == q['gt_letter'])
        rec = {
            'idx': state['idx'],
            'sample_id': q['sample_id'],
            'target': q['target'],
            'picked': letter,
            'gt_letter': q['gt_letter'],
            'gt_face': q['gt_face'],
            'correct': correct,
        }
        state['records'].append(rec)
        if state['show_feedback']:
            tick = '✓ correct' if correct else '✗ wrong'
            feedback_text = (f'**{tick}** — picked **{letter}**, '
                              f'GT was **{q["gt_letter"]}** (face: {q["gt_face"]}).')
        else:
            feedback_text = '_(answer recorded; feedback hidden)_'
        # Advance to next
        state['idx'] += 1
        if state['idx'] >= len(queries):
            n_correct = sum(1 for r in state['records'] if r['correct'])
            n_total = len(state['records'])
            done_msg = (f'### Trial complete!\n\n**{n_correct}/{n_total} = '
                         f'{n_correct/n_total:.1%}**')
            save_trial_log()
            return (done_msg, None, None, 'No more queries.',
                    feedback_text, gr.update(visible=True))
        # Load next
        q2 = load_query(state['idx'])
        status = render_status(state['idx'])
        question = (f'**Target:** *{q2["target"]}*\n\n'
                     f'(Sample {q2["sample_id"]})\n\n'
                     f'Where is the target? Pick A/B/C/D, or Skip.')
        return (status, q2['frame'], q2['cubemap'], question,
                feedback_text, gr.update(visible=False))

    def skip():
        state['idx'] += 1
        if state['idx'] >= len(queries):
            return ('### Done (skipped to end)', None, None,
                    'No more queries.', '_(skipped)_',
                    gr.update(visible=True))
        q = load_query(state['idx'])
        status = render_status(state['idx'])
        question = (f'**Target:** *{q["target"]}*\n\n'
                     f'(Sample {q["sample_id"]})\n\n'
                     f'Where is the target? Pick A/B/C/D, or Skip.')
        return (status, q['frame'], q['cubemap'], question,
                '_(skipped)_', gr.update(visible=False))

    def restart():
        state['idx'] = 0
        state['records'] = []
        return get_initial_view()

    def toggle_feedback(show):
        state['show_feedback'] = show
        return gr.update()

    def save_trial_log():
        if not state['records']:
            return
        import time
        ts = time.strftime('%Y%m%d_%H%M%S')
        path = TRIAL_LOG_DIR / f'trial_{ts}.jsonl'
        with open(path, 'w') as f:
            for r in state['records']:
                f.write(json.dumps(r) + '\n')
        print(f'Saved trial log: {path}')

    with gr.Blocks(title='Q2 Human Trial', theme=gr.themes.Soft()) as app:
        gr.Markdown('# Q2 Human Trial — kitchen object localization\n'
                     'For each query, look at the original frame (left) and the '
                     '2x3 cubemap with A/B/C/D markers (right). Pick the marker '
                     'that you think is closest to the target object\'s actual '
                     '3D location. The target is OUT OF VIEW.')

        with gr.Row():
            status_md = gr.Markdown(value='_(loading)_')
            feedback_toggle = gr.Checkbox(label='Show GT feedback after each pick',
                                            value=True)

        with gr.Row():
            frame_img = gr.Image(label='Original frame (perspective)',
                                  type='pil', height=380)
            cubemap_img = gr.Image(label='2x3 cubemap (markers in dark regions)',
                                     type='pil', height=380)

        question_md = gr.Markdown(value='_(loading)_')
        feedback_md = gr.Markdown(value='_(no answer yet)_')

        with gr.Row():
            btn_a = gr.Button('A', variant='primary')
            btn_b = gr.Button('B', variant='primary')
            btn_c = gr.Button('C', variant='primary')
            btn_d = gr.Button('D', variant='primary')
            btn_skip = gr.Button('Skip', variant='secondary')

        with gr.Row():
            btn_restart = gr.Button('Restart', variant='secondary',
                                       visible=False)

        outputs = [status_md, frame_img, cubemap_img, question_md,
                   feedback_md, btn_restart]

        btn_a.click(fn=lambda: pick_letter('A'), outputs=outputs)
        btn_b.click(fn=lambda: pick_letter('B'), outputs=outputs)
        btn_c.click(fn=lambda: pick_letter('C'), outputs=outputs)
        btn_d.click(fn=lambda: pick_letter('D'), outputs=outputs)
        btn_skip.click(fn=skip, outputs=outputs)
        btn_restart.click(fn=restart, outputs=outputs)
        feedback_toggle.change(fn=toggle_feedback, inputs=[feedback_toggle])

        app.load(fn=get_initial_view, outputs=outputs)

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=100)
    parser.add_argument('--port', type=int, default=17861)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--share', action='store_true')
    args = parser.parse_args()

    app = make_app(args.n, args.seed)
    app.launch(server_name='0.0.0.0', server_port=args.port,
                share=args.share, show_error=True)


if __name__ == '__main__':
    main()
