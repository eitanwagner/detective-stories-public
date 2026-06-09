"""Analysis methods for DetectiveStoryAnalyzer.

These methods are added to the DetectiveStoryAnalyzer class via a mixin pattern.
Import and apply them in main.py.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf
import json

from config import (
    USE_ARGMAX, NAIVE_THRESHOLD, REAL_STORY_S_MODEL, EXPERIENCED_READER_LABELS,
    EXPERIENCED_READER_SHORT_LABELS, EXPERIENCED_READER_COLORS,
    model_names_all, model_names_with_human, display_names,
    EARLY_REVEAL_THRESHOLD, EARLY_REVEAL_PARAGRAPHS,
    MODEL_GEN_SUCCESS, MODEL_N_VALID,
)
from utils import clopper_pearson_interval, argmax_distribution


class AnalysisMixin:
    """Mixin class containing all analysis/reporting methods.

    These methods assume the base class provides:
        - self.human_data, self.llm_data, self.experienced_reader_data
        - self.get_human_distribution(), self.get_llm_distributions()
        - self.get_naive_reader_probs(), self.get_experienced_reader_distributions()
        - self._resolve_llm_key()
        - self.naive_reader_mode
    """

    def print_all_fairplay_scores(self):
        """Print comprehensive fairplay, surprise, and coherence scores for all stories and models"""

        print("\n" + "=" * 80)
        print("COMPREHENSIVE FAIRPLAY, SURPRISE, AND COHERENCE SCORES REPORT")
        print("=" * 80)

        # =====================================================================
        # PART 1: LLM-based fairplay for ALL stories (including without human data)
        # =====================================================================
        print("\n" + "-" * 80)
        print("PART 1: LLM-BASED SCORES (ALL STORIES)")
        print("-" * 80)

        model_names = model_names_all
        saved_mode = self.naive_reader_mode

        # Organize data by model, preferring nsamp=20
        model_data = {}
        for key, value in self.llm_data.items():
            model_name, story_type, story_id = key
            nsamp = value['params'].get('nsamp', '5')
            story_key = f"{model_name}_{story_type}_{story_id}"

            if model_name not in model_data:
                model_data[model_name] = {}

            if story_key not in model_data[model_name] or nsamp == '20':
                model_data[model_name][story_key] = (key, value, nsamp)

        def _print_scores(label, fp_scores, surprise_scores, coherence_scores, story_details,
                          exp_fp_scores=None, exp_coherence_scores=None):
            print(f"\n{label}:")
            # Per-story breakdown
            print(f"  {'story_id':<30}  {'FP':>7}  {'Surprise':>8}  {'Coherence':>9}  {'Cl-Def':>7}")
            print(f"  {'-'*30}  {'-'*7}  {'-'*8}  {'-'*9}  {'-'*7}")
            for row in story_details:
                stype, sid, fp, sur, coh = row[0], row[1], row[2], row[3], row[4]
                diff = row[5] if len(row) > 5 else None
                diff_str = f"{diff:+7.2f}" if diff is not None else "    n/a"
                print(f"  {f'{stype} {sid}':<30}  {fp:7.2f}  {sur:8.2f}  {coh:9.2f}  {diff_str}")
            # Aggregates
            print(f"  LLM Fair Play (x25):  Mean={np.mean(fp_scores):.3f}, Std={np.std(fp_scores):.3f}, n={len(fp_scores)}")
            print(f"  LLM Fair Play:        Mean={np.mean(fp_scores) / 25:.3f}")
            print(f"  LLM Surprise (x25):   Mean={np.mean(surprise_scores):.3f}, Std={np.std(surprise_scores):.3f}")
            print(f"  LLM Surprise:         Mean={np.mean(surprise_scores) / 25:.3f}")
            print(f"  LLM Coherence (x25):  Mean={np.mean(coherence_scores):.3f}, Std={np.std(coherence_scores):.3f}")
            print(f"  LLM Coherence:        Mean={np.mean(coherence_scores) / 25:.3f}")
            num_leq_1 = sum(1 for s in story_details if s[2] <= 1)
            print(f"  Stories with LLM FP <= 1: {num_leq_1}/{len(story_details)} ({num_leq_1 / len(story_details) * 100:.1f}%)")
            diffs = [s[5] for s in story_details if len(s) > 5 and s[5] is not None]
            if diffs:
                n_lt1 = sum(1 for d in diffs if d < 1)
                print(f"  Clueless−Default (x25): Mean={np.mean(diffs):.3f}, Std={np.std(diffs):.3f}")
                print(f"  Stories where Clueless−Default < 1: {n_lt1}/{len(diffs)} ({n_lt1 / len(diffs) * 100:.1f}%)")
            if exp_fp_scores:
                print(f"  Exp FP (x25):         Mean={np.mean(exp_fp_scores):.3f}, Std={np.std(exp_fp_scores):.3f}, n={len(exp_fp_scores)}")
                print(f"  Exp FP:               Mean={np.mean(exp_fp_scores) / 25:.3f}")
                print(f"  Exp Coherence (x25):  Mean={np.mean(exp_coherence_scores):.3f}, Std={np.std(exp_coherence_scores):.3f}")
                print(f"  Exp Coherence:        Mean={np.mean(exp_coherence_scores) / 25:.3f}")

        # Artificial stories: group by the model that wrote them
        print("\nARTIFICIAL STORIES:")
        print("-" * 80)
        for model in model_names:
            if model not in model_data:
                continue

            llm_fp_scores = []
            llm_surprise_scores = []
            llm_coherence_scores = []
            story_details = []
            exp_fp_scores = []
            exp_coherence_scores = []

            for story_key, (key, value, nsamp) in model_data[model].items():
                model_name, story_type, story_id = key

                if story_type in ['poirot', 'sherlock']:
                    continue

                sampling_dist, _, _, sampling_ci, nsamp_data = self.get_llm_distributions(
                    model_name, story_type, story_id
                )
                naive_dist, true_idx = self.get_naive_reader_probs(f"{model_name} {story_id}")
                if naive_dist is None or sampling_dist is None or true_idx is None:
                    continue
                if len(sampling_dist) != len(naive_dist):
                    continue

                naive_true = naive_dist[:, true_idx]
                sampling_true = sampling_dist[:, true_idx]
                llm_fp = np.mean(sampling_true - naive_true)
                llm_fp_scores.append(llm_fp * 25)
                llm_surprise_scores.append(np.mean(naive_true) * 25)
                llm_coherence_scores.append(np.mean(sampling_true) * 25)

                # Clueless−Default diff (independent of current naive mode)
                self.naive_reader_mode = "default"
                naive_def, tidx_def = self.get_naive_reader_probs(f"{model_name} {story_id}")
                self.naive_reader_mode = "clueless"
                naive_cl, tidx_cl = self.get_naive_reader_probs(f"{model_name} {story_id}")
                self.naive_reader_mode = saved_mode
                cl_def_diff = None
                if naive_def is not None and naive_cl is not None and tidx_def is not None and tidx_cl is not None:
                    cl_def_diff = (np.mean(naive_cl[:, tidx_cl]) - np.mean(naive_def[:, tidx_def])) * 25

                story_details.append((story_type, story_id, llm_fp * 25, np.mean(naive_true) * 25, np.mean(sampling_true) * 25, cl_def_diff))

                exp_dist, _ = self.get_experienced_reader_distributions(
                    model_name, story_type, story_id, setting=0
                )
                if exp_dist is not None and len(exp_dist) == len(naive_dist):
                    exp_true = exp_dist[:, true_idx]
                    exp_fp_scores.append(np.mean(exp_true - naive_true) * 25)
                    exp_coherence_scores.append(np.mean(exp_true) * 25)

            if llm_fp_scores:
                _print_scores(model, llm_fp_scores, llm_surprise_scores, llm_coherence_scores, story_details,
                              exp_fp_scores or None, exp_coherence_scores or None)

        # Real stories: group by story type (poirot/sherlock), using REAL_STORY_S_MODEL for sampling.
        # Real stories are NOT attributed to any LLM model since the LLM did not write them.
        print("\nREAL STORIES:")
        print("-" * 80)
        for story_type_label in ['poirot', 'sherlock']:
            llm_fp_scores = []
            llm_surprise_scores = []
            llm_coherence_scores = []
            story_details = []
            exp_fp_scores = []
            exp_coherence_scores = []

            # Collect all real-story entries for this story_type across all sampling models
            # Store (fp, surprise, coherence, naive_true, model_name) per story_id, prefer REAL_STORY_S_MODEL
            seen_story_ids = {}
            for model_key, stories in model_data.items():
                for story_key, (key, value, nsamp) in stories.items():
                    model_name, story_type, story_id = key
                    if story_type != story_type_label:
                        continue
                    sampling_dist, _, _, sampling_ci, nsamp_data = self.get_llm_distributions(
                        model_name, story_type, story_id
                    )
                    naive_dist, true_idx = self.get_naive_reader_probs(f"{story_type} {story_id}")
                    if naive_dist is None or sampling_dist is None or true_idx is None:
                        continue
                    if len(sampling_dist) != len(naive_dist):
                        continue
                    naive_true = naive_dist[:, true_idx]
                    sampling_true = sampling_dist[:, true_idx]
                    fp = np.mean(sampling_true - naive_true) * 25
                    surprise = np.mean(naive_true) * 25
                    coherence = np.mean(sampling_true) * 25

                    # Clueless−Default diff (naive dist for real stories is model-independent)
                    self.naive_reader_mode = "default"
                    naive_def, tidx_def = self.get_naive_reader_probs(f"{story_type} {story_id}")
                    self.naive_reader_mode = "clueless"
                    naive_cl, tidx_cl = self.get_naive_reader_probs(f"{story_type} {story_id}")
                    self.naive_reader_mode = saved_mode
                    cl_def_diff = None
                    if naive_def is not None and naive_cl is not None and tidx_def is not None and tidx_cl is not None:
                        cl_def_diff = (np.mean(naive_cl[:, tidx_cl]) - np.mean(naive_def[:, tidx_def])) * 25

                    # Prefer REAL_STORY_S_MODEL; otherwise keep first seen
                    if story_id not in seen_story_ids or model_name == REAL_STORY_S_MODEL:
                        seen_story_ids[story_id] = (fp, surprise, coherence, naive_true, model_name, cl_def_diff, true_idx)

            for story_id, (fp, surprise, coherence, naive_true, best_model, cl_def_diff, true_idx) in seen_story_ids.items():
                llm_fp_scores.append(fp)
                llm_surprise_scores.append(surprise)
                llm_coherence_scores.append(coherence)
                story_details.append((story_type_label, story_id, fp, surprise, coherence, cl_def_diff))

                exp_dist, _ = self.get_experienced_reader_distributions(
                    best_model, story_type_label, story_id, setting=0
                )
                if exp_dist is not None and len(exp_dist) == len(naive_true):
                    exp_true = exp_dist[:, true_idx]
                    exp_fp_scores.append(np.mean(exp_true - naive_true) * 25)
                    exp_coherence_scores.append(np.mean(exp_true) * 25)

            if llm_fp_scores:
                _print_scores(story_type_label, llm_fp_scores, llm_surprise_scores, llm_coherence_scores, story_details,
                              exp_fp_scores or None, exp_coherence_scores or None)

        # =====================================================================
        # PART 2: Human-based scores (only stories with human data)
        # =====================================================================
        print("\n" + "-" * 80)
        print("PART 2: HUMAN-BASED SCORES (STORIES WITH HUMAN DATA)")
        print("     Also showing LLM scores for the same stories")
        print("-" * 80)

        # Organize by model
        human_scores_by_model = {}

        for story_name in self.human_data.keys():
            model_key, story_id = story_name.split()

            # Get human distributions
            human_dist, _, num_responses = self.get_human_distribution(story_name)
            if human_dist is None:
                continue

            naive_dist, true_idx = self.get_naive_reader_probs(story_name)
            if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                continue

            human_true = human_dist[:, true_idx]
            naive_true = naive_dist[:, true_idx]

            # Human scores
            human_fp = np.mean(human_true - naive_true) * 25
            human_surprise = np.mean(naive_true) * 25
            human_coherence = np.mean(human_true) * 25

            # Get LLM distributions for the same story
            is_real = model_key in ['poirot', 'sherlock']
            if is_real:
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    REAL_STORY_S_MODEL, model_key, story_id
                )
            else:
                model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    model_key, model_key2, story_id
                )

            # LLM scores (if available)
            llm_fp = None
            llm_coherence = None
            if sampling_dist is not None and len(sampling_dist) == len(human_dist):
                sampling_true = sampling_dist[:, true_idx]
                llm_fp = np.mean(sampling_true - naive_true) * 25
                llm_coherence = np.mean(sampling_true) * 25

            # Categorize as artificial or real
            category = "real" if is_real else "artificial"

            if model_key not in human_scores_by_model:
                human_scores_by_model[model_key] = {'artificial': [], 'real': []}

            human_scores_by_model[model_key][category].append({
                'story_id': story_id,
                'human_fp': human_fp,
                'human_surprise': human_surprise,
                'human_coherence': human_coherence,
                'llm_fp': llm_fp,
                'llm_coherence': llm_coherence,
                'num_responses': num_responses
            })

        # Print results by category
        for story_category in ["ARTIFICIAL STORIES", "REAL STORIES"]:
            print(f"\n{story_category}:")
            print("-" * 80)

            category_key = "artificial" if story_category == "ARTIFICIAL STORIES" else "real"

            for model in sorted(human_scores_by_model.keys()):
                stories = human_scores_by_model[model][category_key]
                if len(stories) == 0:
                    continue

                # Human scores
                human_fp_scores = [s['human_fp'] for s in stories]
                human_surprise_scores = [s['human_surprise'] for s in stories]
                human_coherence_scores = [s['human_coherence'] for s in stories]

                # LLM scores (only for stories where available)
                llm_fp_scores = [s['llm_fp'] for s in stories if s['llm_fp'] is not None]
                llm_coherence_scores = [s['llm_coherence'] for s in stories if s['llm_coherence'] is not None]

                print(f"\n{model}:")
                print(f"  HUMAN SCORES:")
                print(
                    f"    Fair Play (x25):  Mean={np.mean(human_fp_scores):.3f}, Std={np.std(human_fp_scores):.3f}, n={len(human_fp_scores)}")
                print(f"    Fair Play:        Mean={np.mean(human_fp_scores) / 25:.3f}")
                print(
                    f"    Surprise (x25):   Mean={np.mean(human_surprise_scores):.3f}, Std={np.std(human_surprise_scores):.3f}")
                print(f"    Surprise:         Mean={np.mean(human_surprise_scores) / 25:.3f}")
                print(
                    f"    Coherence (x25):  Mean={np.mean(human_coherence_scores):.3f}, Std={np.std(human_coherence_scores):.3f}")
                print(f"    Coherence:        Mean={np.mean(human_coherence_scores) / 25:.3f}")

                if len(llm_fp_scores) > 0:
                    print(f"  LLM SCORES (same stories):")
                    print(
                        f"    Fair Play (x25):  Mean={np.mean(llm_fp_scores):.3f}, Std={np.std(llm_fp_scores):.3f}, n={len(llm_fp_scores)}")
                    print(f"    Fair Play:        Mean={np.mean(llm_fp_scores) / 25:.3f}")
                    print(
                        f"    Coherence (x25):  Mean={np.mean(llm_coherence_scores):.3f}, Std={np.std(llm_coherence_scores):.3f}")
                    print(f"    Coherence:        Mean={np.mean(llm_coherence_scores) / 25:.3f}")

                # Statistics
                num_leq_1_human = sum(1 for s in stories if s['human_fp'] <= 1)
                print(
                    f"  Stories with Human FP <= 1: {num_leq_1_human}/{len(stories)} ({num_leq_1_human / len(stories) * 100:.1f}%)")
                if len(llm_fp_scores) > 0:
                    num_leq_1_llm = sum(1 for s in stories if s['llm_fp'] is not None and s['llm_fp'] <= 1)
                    print(
                        f"  Stories with LLM FP <= 1: {num_leq_1_llm}/{len(llm_fp_scores)} ({num_leq_1_llm / len(llm_fp_scores) * 100:.1f}%)")

        # =====================================================================
        # PART 3: Combined comparison for stories with both human and LLM data
        # =====================================================================
        print("\n" + "-" * 80)
        print("PART 3: HUMAN vs LLM COMPARISON (STORIES WITH BOTH)")
        print("-" * 80)

        comparison_data = []

        for story_name in self.human_data.keys():
            model_key, story_id = story_name.split()

            # Get human fairplay
            human_dist, _, num_responses = self.get_human_distribution(story_name)
            if human_dist is None:
                continue

            naive_dist, true_idx = self.get_naive_reader_probs(story_name)
            if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                continue

            human_true = human_dist[:, true_idx]
            naive_true = naive_dist[:, true_idx]

            # Human scores
            human_fp = np.mean(human_true - naive_true) * 25
            human_surprise = np.mean(naive_true) * 25
            human_coherence = np.mean(human_true) * 25

            # Get LLM fairplay
            is_real = model_key in ['poirot', 'sherlock']
            if is_real:
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    REAL_STORY_S_MODEL, model_key, story_id
                )
            else:
                model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    model_key, model_key2, story_id
                )

            if sampling_dist is None or len(sampling_dist) != len(human_dist):
                continue

            sampling_true = sampling_dist[:, true_idx]

            # LLM scores
            llm_fp = np.mean(sampling_true - naive_true) * 25
            llm_surprise = np.mean(naive_true) * 25  # Same as human surprise
            llm_coherence = np.mean(sampling_true) * 25

            comparison_data.append({
                'story': story_name,
                'is_real': is_real,
                'human_fp': human_fp,
                'llm_fp': llm_fp,
                'fp_diff': human_fp - llm_fp,
                'human_surprise': human_surprise,
                'llm_surprise': llm_surprise,
                'surprise_diff': human_surprise - llm_surprise,
                'human_coherence': human_coherence,
                'llm_coherence': llm_coherence,
                'coherence_diff': human_coherence - llm_coherence,
                'num_responses': num_responses
            })

        # Print comparison by category
        for story_category, is_real in [("ARTIFICIAL STORIES", False), ("REAL STORIES", True)]:
            print(f"\n{story_category}:")
            print("-" * 80)

            category_data = [d for d in comparison_data if d['is_real'] == is_real]
            if len(category_data) == 0:
                continue

            human_fp = [d['human_fp'] for d in category_data]
            llm_fp = [d['llm_fp'] for d in category_data]
            fp_diffs = [d['fp_diff'] for d in category_data]

            human_surprise = [d['human_surprise'] for d in category_data]
            llm_surprise = [d['llm_surprise'] for d in category_data]
            surprise_diffs = [d['surprise_diff'] for d in category_data]

            human_coherence = [d['human_coherence'] for d in category_data]
            llm_coherence = [d['llm_coherence'] for d in category_data]
            coherence_diffs = [d['coherence_diff'] for d in category_data]

            print(f"\nAggregate Statistics (n={len(category_data)}):")
            print(f"\n  FAIR PLAY:")
            print(f"    Human (x25):      Mean={np.mean(human_fp):.3f}, Std={np.std(human_fp):.3f}")
            print(f"    Human:            Mean={np.mean(human_fp) / 25:.3f}")
            print(f"    LLM (x25):        Mean={np.mean(llm_fp):.3f}, Std={np.std(llm_fp):.3f}")
            print(f"    LLM:              Mean={np.mean(llm_fp) / 25:.3f}")
            print(f"    Difference (x25): Mean={np.mean(fp_diffs):.3f}, Std={np.std(fp_diffs):.3f}")
            print(f"    Difference:       Mean={np.mean(fp_diffs) / 25:.3f}")

            print(f"\n  SURPRISE:")
            print(f"    Naive (x25):      Mean={np.mean(human_surprise):.3f}, Std={np.std(human_surprise):.3f}")
            print(f"    Naive:            Mean={np.mean(human_surprise) / 25:.3f}")

            print(f"\n  COHERENCE:")
            print(f"    Human (x25):      Mean={np.mean(human_coherence):.3f}, Std={np.std(human_coherence):.3f}")
            print(f"    Human:            Mean={np.mean(human_coherence) / 25:.3f}")
            print(f"    LLM (x25):        Mean={np.mean(llm_coherence):.3f}, Std={np.std(llm_coherence):.3f}")
            print(f"    LLM:              Mean={np.mean(llm_coherence) / 25:.3f}")
            print(f"    Difference (x25): Mean={np.mean(coherence_diffs):.3f}, Std={np.std(coherence_diffs):.3f}")
            print(f"    Difference:       Mean={np.mean(coherence_diffs) / 25:.3f}")

            # Group stories by model
            stories_by_model = {}
            for d in category_data:
                model = d['story'].split()[0]
                if model not in stories_by_model:
                    stories_by_model[model] = []
                stories_by_model[model].append(d)

            # Per-model comparisons
            print(f"\n  PER-MODEL COMPARISONS:")

            for model in sorted(stories_by_model.keys()):
                model_stories = stories_by_model[model]
                print(f"\n  {model}:")

                # Stories with LLM FP <= 1
                llm_fp_leq_1 = [d for d in model_stories if d['llm_fp'] <= 1]
                print(
                    f"    Stories with LLM FP <= 1: {len(llm_fp_leq_1)}/{len(model_stories)} ({len(llm_fp_leq_1) / len(model_stories) * 100:.1f}%)")
                if llm_fp_leq_1:
                    for d in sorted(llm_fp_leq_1, key=lambda x: x['llm_fp']):
                        print(f"      {d['story']:<30} LLM FP={d['llm_fp']:>6.3f}")

                # Stories where LLM FP is within 1 point of Human FP
                llm_within_1 = [d for d in model_stories if d['fp_diff'] >= -1]
                print(
                    f"    Stories where LLM FP is within 1 point of Human FP: {len(llm_within_1)}/{len(model_stories)} ({len(llm_within_1) / len(model_stories) * 100:.1f}%)")
                if llm_within_1:
                    for d in sorted(llm_within_1, key=lambda x: x['fp_diff'], reverse=True):
                        print(
                            f"      {d['story']:<30} Human={d['human_fp']:>6.3f}, LLM={d['llm_fp']:>6.3f}, Diff={d['fp_diff']:>6.3f}")

                # Stories where Human FP is better than LLM FP by >1 point
                human_better = [d for d in model_stories if d['fp_diff'] > 1]
                print(
                    f"    Stories where Human FP is better than LLM FP by >1 point: {len(human_better)}/{len(model_stories)} ({len(human_better) / len(model_stories) * 100:.1f}%)")
                if human_better:
                    for d in sorted(human_better, key=lambda x: x['fp_diff'], reverse=True):
                        print(
                            f"      {d['story']:<30} Human={d['human_fp']:>6.3f}, LLM={d['llm_fp']:>6.3f}, Diff={d['fp_diff']:>6.3f}")

        # =====================================================================
        # PART 4: AGGREGATE SCORES (per-story STD)
        # =====================================================================
        print("\n" + "-" * 80)
        print("PART 4: AGGREGATE SCORES (Mean ± SD over individual stories)")
        print("  STD is computed over per-story scores, NOT average of averages")
        print("-" * 80)

        # Collect per-story scores for all stories with human data
        all_story_scores = []
        for d in comparison_data:
            all_story_scores.append(d)

        if len(all_story_scores) > 0:
            artificial_stories = [d for d in all_story_scores if not d['is_real']]
            real_stories = [d for d in all_story_scores if d['is_real']]

            for label, subset in [("ALL STORIES", all_story_scores),
                                   ("ARTIFICIAL STORIES", artificial_stories),
                                   ("REAL STORIES", real_stories)]:
                if len(subset) == 0:
                    continue
                print(f"\n  {label} (n={len(subset)}):")

                human_fp_vals = [d['human_fp'] for d in subset]
                llm_fp_vals = [d['llm_fp'] for d in subset]
                human_surprise_vals = [d['human_surprise'] for d in subset]
                human_coherence_vals = [d['human_coherence'] for d in subset]
                llm_coherence_vals = [d['llm_coherence'] for d in subset]

                print(f"    Human Fair Play (x25):   {np.mean(human_fp_vals):.3f} ± {np.std(human_fp_vals):.3f}")
                print(f"    Human Fair Play:         {np.mean(human_fp_vals) / 25:.3f} ± {np.std(human_fp_vals) / 25:.3f}")
                print(f"    LLM Fair Play (x25):     {np.mean(llm_fp_vals):.3f} ± {np.std(llm_fp_vals):.3f}")
                print(f"    LLM Fair Play:           {np.mean(llm_fp_vals) / 25:.3f} ± {np.std(llm_fp_vals) / 25:.3f}")
                print(f"    Surprise (x25):          {np.mean(human_surprise_vals):.3f} ± {np.std(human_surprise_vals):.3f}")
                print(f"    Surprise:                {np.mean(human_surprise_vals) / 25:.3f} ± {np.std(human_surprise_vals) / 25:.3f}")
                print(f"    Human Coherence (x25):   {np.mean(human_coherence_vals):.3f} ± {np.std(human_coherence_vals):.3f}")
                print(f"    Human Coherence:         {np.mean(human_coherence_vals) / 25:.3f} ± {np.std(human_coherence_vals) / 25:.3f}")
                print(f"    LLM Coherence (x25):     {np.mean(llm_coherence_vals):.3f} ± {np.std(llm_coherence_vals):.3f}")
                print(f"    LLM Coherence:           {np.mean(llm_coherence_vals) / 25:.3f} ± {np.std(llm_coherence_vals) / 25:.3f}")

        print("\n" + "=" * 80)
        print("END OF COMPREHENSIVE SCORES REPORT")
        print("=" * 80 + "\n")


    def analyze_participant_variance(self):
        """Analyze variance of participant performance as a function of story"""

        story_variances = []

        for story_name, data in self.human_data.items():
            df = data['df']
            response_cols = data['response_cols']

            # Get true culprit from majority vote at end
            final_responses = df[response_cols[-1]].dropna()
            if len(final_responses) == 0:
                continue

            true_culprit = final_responses.mode()[0] if len(final_responses.mode()) > 0 else None
            if true_culprit is None:
                continue

            # Calculate accuracy for each participant
            participant_accuracies = []

            for idx, row in df.iterrows():
                correct_count = 0
                total_count = 0

                for col in response_cols:
                    if pd.notna(row[col]):
                        total_count += 1
                        if row[col] == true_culprit:
                            correct_count += 1

                if total_count > 0:
                    accuracy = correct_count / total_count
                    participant_accuracies.append(accuracy)

            if len(participant_accuracies) > 1:
                story_variances.append({
                    'story': story_name,
                    'variance': np.var(participant_accuracies),
                    'std': np.std(participant_accuracies),
                    'mean_accuracy': np.mean(participant_accuracies),
                    'n_participants': len(participant_accuracies),
                    'min_accuracy': np.min(participant_accuracies),
                    'max_accuracy': np.max(participant_accuracies),
                    'range': np.max(participant_accuracies) - np.min(participant_accuracies)
                })

        variance_df = pd.DataFrame(story_variances)

        # Print summary
        print("\n=== PARTICIPANT PERFORMANCE VARIANCE BY STORY ===\n")
        print(variance_df.to_string(index=False))

        print("\n=== SUMMARY STATISTICS ===")
        print(f"Average variance across stories: {variance_df['variance'].mean():.4f}")
        print(f"Average std across stories: {variance_df['std'].mean():.4f}")
        print(f"Average range across stories: {variance_df['range'].mean():.4f}")

        # Plot variance by story
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Plot 1: Variance and std by story
        x = np.arange(len(variance_df))
        axes[0].bar(x - 0.2, variance_df['variance'], 0.4, label='Variance', alpha=0.8)
        axes[0].bar(x + 0.2, variance_df['std'], 0.4, label='Std Dev', alpha=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(variance_df['story'], rotation=45, ha='right')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Participant Performance Variance by Story')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)

        # Plot 2: Range (max - min accuracy) by story
        axes[1].bar(x, variance_df['range'], alpha=0.8, color='coral')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(variance_df['story'], rotation=45, ha='right')
        axes[1].set_ylabel('Accuracy Range')
        axes[1].set_title('Range of Participant Accuracies by Story')
        axes[1].grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig('participant_variance_by_story.png', dpi=300, bbox_inches='tight')
        plt.show()

        return variance_df


    def analyze_subjective_ratings(self):
        """
        Analyze correlation between subjective ratings and computed metrics
        using INDIVIDUAL-LEVEL data instead of story-level averages.

        Returns:
            dict: Results containing correlations at multiple levels
            pd.DataFrame: Individual-level data for further analysis
        """

        results = {
            'individual_level': {},  # Correlations using all individual responses
            'story_level': {},  # Original story-level for comparison
            'per_model': {},  # Correlations within each model
            'sample_sizes': {}  # Track how many observations we have
        }

        # Collect individual-level data
        individual_data = []
        story_aggregated_data = []

        for story_name, data in self.human_data.items():
            df = data['df']
            ratings = data['ratings']

            if len(ratings) == 0:
                continue

            # Get computational measures for this story
            human_dist, _, _ = self.get_human_distribution(story_name)
            if human_dist is None:
                continue

            naive_dist, true_idx = self.get_naive_reader_probs(story_name)
            if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                continue

            model_key, story_id = story_name.split()
            is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

            # Get sampling distribution
            if not is_artificial:
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    "gemini-1.5-flash", model_key, story_id
                )
            else:
                model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    model_key, model_key2, story_id
                )

            if sampling_dist is None or len(sampling_dist) != len(human_dist):
                continue

            # Calculate story-level metrics (for the true culprit trajectory)
            human_true = human_dist[:, true_idx]
            naive_true = naive_dist[:, true_idx]
            sampling_true = sampling_dist[:, true_idx]

            # Story-level computational measures
            surprise_metric = 1 - np.mean(naive_true)
            coherence_knowitall = np.mean(sampling_true)
            coherence_human = np.mean(human_true)
            fairplay_human = np.mean(human_true - naive_true)
            fairplay_llm = np.mean(sampling_true - naive_true)

            # Extract rating column names
            rating_cols = df.columns[-6:]

            # INDIVIDUAL LEVEL: Loop through each participant
            for idx, row in df.iterrows():
                # FIX: Use actual user ID from column 1, not row index!
                participant_id = row[df.columns[1]]

                participant_ratings = {}

                # Extract this participant's ratings
                for col in rating_cols:
                    col_lower = col.lower()
                    if 'familiarity' in col_lower:
                        participant_ratings['familiarity'] = row[col]
                    elif 'effort' in col_lower:
                        participant_ratings['effort'] = row[col]
                    elif 'surprising' in col_lower:
                        participant_ratings['surprise'] = row[col]
                    elif 'fair' in col_lower:
                        participant_ratings['fairness'] = row[col]
                    elif 'coherent' in col_lower:
                        participant_ratings['coherence'] = row[col]
                    elif 'enjoyable' in col_lower:
                        participant_ratings['enjoyability'] = row[col]

                # Skip if participant didn't provide ratings
                if not participant_ratings:
                    continue

                # Add row for this participant
                individual_data.append({
                    'participant_id': participant_id,
                    'story': story_name,
                    'model': model_key,
                    'is_artificial': is_artificial,
                    **participant_ratings,
                    # Computational measures (same for all participants in this story)
                    'surprise_metric': surprise_metric,
                    'coherence_knowitall': coherence_knowitall,
                    'coherence_human': coherence_human,
                    'fairplay_human': fairplay_human,
                    'fairplay_llm': fairplay_llm
                })

            # Also collect story-level aggregated data (for comparison)
            avg_ratings = {}
            for col in rating_cols:
                col_lower = col.lower()
                if 'familiarity' in col_lower:
                    avg_ratings['familiarity'] = df[col].mean()
                elif 'effort' in col_lower:
                    avg_ratings['effort'] = df[col].mean()
                elif 'surprising' in col_lower:
                    avg_ratings['surprise'] = df[col].mean()
                elif 'fair' in col_lower:
                    avg_ratings['fairness'] = df[col].mean()
                elif 'coherent' in col_lower:
                    avg_ratings['coherence'] = df[col].mean()
                elif 'enjoyable' in col_lower:
                    avg_ratings['enjoyability'] = df[col].mean()

            story_aggregated_data.append({
                'story': story_name,
                'model': model_key,
                **avg_ratings,
                'surprise_metric': surprise_metric,
                'coherence_knowitall': coherence_knowitall,
                'coherence_human': coherence_human,
                'fairplay_human': fairplay_human,
                'fairplay_llm': fairplay_llm
            })

        # Create DataFrames
        df_individual = pd.DataFrame(individual_data)
        df_stories = pd.DataFrame(story_aggregated_data)

        # === INDIVIDUAL-LEVEL CORRELATIONS ===
        print("\n" + "=" * 70)
        print("INDIVIDUAL-LEVEL CORRELATIONS (using all participant responses)")
        print("=" * 70 + "\n")

        individual_correlations = {}

        if 'surprise' in df_individual.columns:
            valid_data = df_individual[['surprise', 'surprise_metric']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['surprise'], valid_data['surprise_metric'])
                individual_correlations['surprise'] = (corr, p)
                print(f"Surprise rating vs Surprise metric:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)}")

        if 'coherence' in df_individual.columns:
            valid_data = df_individual[['coherence', 'coherence_knowitall']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['coherence'], valid_data['coherence_knowitall'])
                individual_correlations['coherence_knowitall'] = (corr, p)
                print(f"\nCoherence rating vs Know-it-all score:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)}")

            valid_data = df_individual[['coherence', 'coherence_human']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['coherence'], valid_data['coherence_human'])
                individual_correlations['coherence_human'] = (corr, p)
                print(f"Coherence rating vs Human score:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)}")

        if 'fairness' in df_individual.columns:
            valid_data = df_individual[['fairness', 'coherence_knowitall']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['fairness'], valid_data['coherence_knowitall'])
                individual_correlations['coherence_fairness_llm'] = (corr, p)
                print(f"\nCoherence metric (llm) vs fairness:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)}")

            valid_data = df_individual[['fairness', 'coherence_human']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['fairness'], valid_data['coherence_human'])
                individual_correlations['coherence_fairness_human'] = (corr, p)
                print(f"Coherence metric (human) vs fairness:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)}")

        if 'fairness' in df_individual.columns:
            valid_data = df_individual[['fairness', 'fairplay_human']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['fairness'], valid_data['fairplay_human'])
                individual_correlations['fairness_human'] = (corr, p)
                print(f"\nFairness rating vs Human-based fair play:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)}")

            valid_data = df_individual[['fairness', 'fairplay_llm']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['fairness'], valid_data['fairplay_llm'])
                individual_correlations['fairness_llm'] = (corr, p)
                print(f"Fairness rating vs LLM-based fair play:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)}")

        if 'enjoyability' in df_individual.columns:
            valid_data = df_individual[['enjoyability', 'fairplay_human']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['enjoyability'], valid_data['fairplay_human'])
                individual_correlations['enjoyability_human'] = (corr, p)
                print(f"\nEnjoyability vs Human-based fair play:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)}")

            valid_data = df_individual[['enjoyability', 'fairplay_llm']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['enjoyability'], valid_data['fairplay_llm'])
                individual_correlations['enjoyability_llm'] = (corr, p)
                print(f"Enjoyability vs LLM-based fair play:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)}")

        results['individual_level'] = individual_correlations

        # === STORY-LEVEL CORRELATIONS (for comparison) ===
        print("\n" + "=" * 70)
        print("STORY-LEVEL CORRELATIONS (for comparison - using story averages)")
        print("=" * 70 + "\n")

        story_correlations = {}

        if 'surprise' in df_stories.columns:
            valid_data = df_stories[['surprise', 'surprise_metric']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['surprise'], valid_data['surprise_metric'])
                story_correlations['surprise'] = (corr, p)
                print(f"Surprise rating vs Surprise metric:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)} stories")

        if 'fairness' in df_stories.columns:
            valid_data = df_stories[['fairness', 'fairplay_human']].dropna()
            if len(valid_data) > 2:
                corr, p = stats.spearmanr(valid_data['fairness'], valid_data['fairplay_human'])
                story_correlations['fairness_human'] = (corr, p)
                print(f"\nFairness rating vs Human-based fair play:")
                print(f"  r = {corr:.3f}, p = {p:.4f}, n = {len(valid_data)} stories")

        results['story_level'] = story_correlations

        # === PER-MODEL CORRELATIONS (individual level) ===
        print("\n" + "=" * 70)
        print("PER-MODEL CORRELATIONS (individual-level)")
        print("=" * 70 + "\n")

        per_model_corr = {}
        for model in df_individual['model'].unique():
            df_model = df_individual[df_individual['model'] == model]

            if len(df_model) < 10:  # Need reasonable sample size
                continue

            print(f"\n{model} (n={len(df_model)} participants):")
            model_corr = {}

            if 'surprise' in df_model.columns:
                valid_data = df_model[['surprise', 'surprise_metric']].dropna()
                if len(valid_data) > 2:
                    corr, p = stats.spearmanr(valid_data['surprise'], valid_data['surprise_metric'])
                    model_corr['surprise'] = (corr, p)
                    print(f"  Surprise: r={corr:.3f}, p={p:.4f}, n={len(valid_data)}")

            if 'fairness' in df_model.columns:
                valid_data = df_model[['fairness', 'fairplay_human']].dropna()
                if len(valid_data) > 2:
                    corr, p = stats.spearmanr(valid_data['fairness'], valid_data['fairplay_human'])
                    model_corr['fairness_human'] = (corr, p)
                    print(f"  Fairness (human): r={corr:.3f}, p={p:.4f}, n={len(valid_data)}")

            per_model_corr[model] = model_corr

        results['per_model'] = per_model_corr

        # Sample size information
        results['sample_sizes'] = {
            'total_participants': len(df_individual),
            'total_stories': len(df_stories),
            'participants_per_story': df_individual.groupby('story').size().to_dict()
        }

        print("\n" + "=" * 70)
        print("SAMPLE SIZE SUMMARY")
        print("=" * 70)
        print(f"Total individual responses: {len(df_individual)}")
        print(f"Total stories: {len(df_stories)}")
        print(f"Average participants per story: {len(df_individual) / len(df_stories):.1f}")

        return results, df_individual, df_stories


    def analyze_subjective_ratings2(self):
        """Analyze correlation between subjective ratings and computed metrics"""

        results = {
            'story_level': {},
            'per_story': {}
        }

        story_data = []

        for story_name, data in self.human_data.items():
            df = data['df']
            ratings = data['ratings']

            if len(ratings) == 0:
                continue

            rating_cols = df.columns[-6:]
            avg_ratings = {}
            for col in rating_cols:
                col_lower = col.lower()
                if 'familiarity' in col_lower:
                    avg_ratings['familiarity'] = df[col].mean()
                elif 'effort' in col_lower:
                    avg_ratings['effort'] = df[col].mean()
                elif 'surprising' in col_lower:
                    avg_ratings['surprise'] = df[col].mean()
                elif 'fair' in col_lower:
                    avg_ratings['fairness'] = df[col].mean()
                elif 'coherent' in col_lower:
                    avg_ratings['coherence'] = df[col].mean()
                elif 'enjoyable' in col_lower:
                    avg_ratings['enjoyability'] = df[col].mean()

            human_dist, _, _ = self.get_human_distribution(story_name)
            if human_dist is None:
                continue

            naive_dist, true_idx = self.get_naive_reader_probs(story_name)
            if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                continue

            model_key, story_id = story_name.split()
            is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

            if not is_artificial:
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    "gemini-1.5-flash", model_key, story_id
                )
            else:
                model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    model_key, model_key2, story_id
                )

            if sampling_dist is None or len(sampling_dist) != len(human_dist):
                continue

            human_true = human_dist[:, true_idx]
            naive_true = naive_dist[:, true_idx]
            sampling_true = sampling_dist[:, true_idx]

            surprise_metric = 1 - np.mean(naive_true)
            coherence_knowitall = np.mean(sampling_true)
            coherence_human = np.mean(human_true)
            fairplay_human = np.mean(human_true - naive_true)
            fairplay_llm = np.mean(sampling_true - naive_true)

            story_data.append({
                'story': story_name,
                'model': model_key,
                **avg_ratings,
                'surprise_metric': surprise_metric,
                'coherence_knowitall': coherence_knowitall,
                'coherence_human': coherence_human,
                'fairplay_human': fairplay_human,
                'fairplay_llm': fairplay_llm
            })

        df_stories = pd.DataFrame(story_data)

        print("\n=== STORY-LEVEL CORRELATIONS (across all stories) ===\n")

        correlations = {}

        if 'surprise' in df_stories.columns:
            corr_surprise, p_surprise = stats.spearmanr(
                df_stories['surprise'].dropna(),
                df_stories['surprise_metric'].loc[df_stories['surprise'].dropna().index]
            )
            correlations['surprise'] = (corr_surprise, p_surprise)
            print(f"Surprise rating vs Surprise metric: r={corr_surprise:.3f}, p={p_surprise:.4f}")

        if 'coherence' in df_stories.columns:
            corr_coh_kit, p_coh_kit = stats.spearmanr(
                df_stories['coherence'].dropna(),
                df_stories['coherence_knowitall'].loc[df_stories['coherence'].dropna().index]
            )
            correlations['coherence_knowitall'] = (corr_coh_kit, p_coh_kit)
            print(f"Coherence rating vs Know-it-all score: r={corr_coh_kit:.3f}, p={p_coh_kit:.4f}")

            corr_coh_human, p_coh_human = stats.spearmanr(
                df_stories['coherence'].dropna(),
                df_stories['coherence_human'].loc[df_stories['coherence'].dropna().index]
            )
            correlations['coherence_human'] = (corr_coh_human, p_coh_human)
            print(f"Coherence rating vs Human score: r={corr_coh_human:.3f}, p={p_coh_human:.4f}")

        if 'fairness' in df_stories.columns:
            corr_fair_human, p_fair_human = stats.spearmanr(
                df_stories['fairness'].dropna(),
                df_stories['fairplay_human'].loc[df_stories['fairness'].dropna().index]
            )
            correlations['fairness_human'] = (corr_fair_human, p_fair_human)
            print(f"Fairness rating vs Human-based fair play: r={corr_fair_human:.3f}, p={p_fair_human:.4f}")

            corr_fair_llm, p_fair_llm = stats.spearmanr(
                df_stories['fairness'].dropna(),
                df_stories['fairplay_llm'].loc[df_stories['fairness'].dropna().index]
            )
            correlations['fairness_llm'] = (corr_fair_llm, p_fair_llm)
            print(f"Fairness rating vs LLM-based fair play: r={corr_fair_llm:.3f}, p={p_fair_llm:.4f}")

        if 'enjoyability' in df_stories.columns:
            corr_enjoy_human, p_enjoy_human = stats.spearmanr(
                df_stories['enjoyability'].dropna(),
                df_stories['fairplay_human'].loc[df_stories['enjoyability'].dropna().index]
            )
            correlations['enjoyability_human'] = (corr_enjoy_human, p_enjoy_human)
            print(f"Enjoyability vs Human-based fair play: r={corr_enjoy_human:.3f}, p={p_enjoy_human:.4f}")

            corr_enjoy_llm, p_enjoy_llm = stats.spearmanr(
                df_stories['enjoyability'].dropna(),
                df_stories['fairplay_llm'].loc[df_stories['enjoyability'].dropna().index]
            )
            correlations['enjoyability_llm'] = (corr_enjoy_llm, p_enjoy_llm)
            print(f"Enjoyability vs LLM-based fair play: r={corr_enjoy_llm:.3f}, p={p_enjoy_llm:.4f}")

        results['story_level'] = correlations

        print("\n=== PER-MODEL CORRELATIONS ===\n")

        per_model_corr = {}
        for model in df_stories['model'].unique():
            df_model = df_stories[df_stories['model'] == model]

            if len(df_model) < 3:
                continue

            print(f"\n{model}:")
            model_corr = {}

            if 'surprise' in df_model.columns and df_model['surprise'].notna().sum() > 2:
                corr, p = stats.spearmanr(df_model['surprise'].dropna(),
                                          df_model['surprise_metric'].loc[df_model['surprise'].dropna().index])
                model_corr['surprise'] = (corr, p)
                print(f"  Surprise: r={corr:.3f}, p={p:.4f}")

            if 'coherence' in df_model.columns and df_model['coherence'].notna().sum() > 2:
                corr_kit, p_kit = stats.spearmanr(df_model['coherence'].dropna(),
                                                  df_model['coherence_knowitall'].loc[
                                                      df_model['coherence'].dropna().index])
                model_corr['coherence_knowitall'] = (corr_kit, p_kit)
                print(f"  Coherence (know-it-all): r={corr_kit:.3f}, p={p_kit:.4f}")

                corr_h, p_h = stats.spearmanr(df_model['coherence'].dropna(),
                                              df_model['coherence_human'].loc[df_model['coherence'].dropna().index])
                model_corr['coherence_human'] = (corr_h, p_h)
                print(f"  Coherence (human): r={corr_h:.3f}, p={p_h:.4f}")

            if 'fairness' in df_model.columns and df_model['fairness'].notna().sum() > 2:
                corr_fh, p_fh = stats.spearmanr(df_model['fairness'].dropna(),
                                                df_model['fairplay_human'].loc[df_model['fairness'].dropna().index])
                model_corr['fairness_human'] = (corr_fh, p_fh)
                print(f"  Fairness (human-based): r={corr_fh:.3f}, p={p_fh:.4f}")

                corr_fl, p_fl = stats.spearmanr(df_model['fairness'].dropna(),
                                                df_model['fairplay_llm'].loc[df_model['fairness'].dropna().index])
                model_corr['fairness_llm'] = (corr_fl, p_fl)
                print(f"  Fairness (LLM-based): r={corr_fl:.3f}, p={p_fl:.4f}")

            per_model_corr[model] = model_corr

        results['per_story'] = per_model_corr

        return results, df_stories


    def analyze_with_mixed_effects(self):
        """
        Analyze subjective ratings using mixed effects models.
        This properly accounts for the nested structure of your data.

        Returns:
        --------
        dict: Results containing:
            - model fits
            - coefficients and p-values
            - ICC (intraclass correlation)
            - comparison with simple methods
        pd.DataFrame: Individual-level data used
        """

        # First, get individual-level data
        print("\n" + "=" * 70)
        print("PREPARING DATA FOR MIXED EFFECTS ANALYSIS")
        print("=" * 70 + "\n")

        individual_data = []

        for story_name, data in self.human_data.items():
            df = data['df']
            ratings = data['ratings']

            if len(ratings) == 0:
                continue

            # Get computational measures
            human_dist, _, _ = self.get_human_distribution(story_name)
            if human_dist is None:
                continue

            naive_dist, true_idx = self.get_naive_reader_probs(story_name)
            if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                continue

            model_key, story_id = story_name.split()
            is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

            # Get sampling distribution
            if not is_artificial:
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    "gemini-1.5-flash", model_key, story_id
                )
            else:
                model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    model_key, model_key2, story_id
                )

            if sampling_dist is None or len(sampling_dist) != len(human_dist):
                continue

            # Calculate computational measures
            human_true = human_dist[:, true_idx]
            naive_true = naive_dist[:, true_idx]
            sampling_true = sampling_dist[:, true_idx]

            surprise_metric = 1 - np.mean(naive_true)
            coherence_knowitall = np.mean(sampling_true)
            coherence_human = np.mean(human_true)
            fairplay_human = np.mean(human_true - naive_true)
            fairplay_llm = np.mean(sampling_true - naive_true)

            # Extract rating columns
            rating_cols = df.columns[-6:]

            # Loop through each participant
            for idx, row in df.iterrows():
                # FIX: Use actual user ID from column 1, not row index!
                participant_id = row[df.columns[1]]

                participant_ratings = {}

                for col in rating_cols:
                    col_lower = col.lower()
                    if 'familiarity' in col_lower:
                        participant_ratings['familiarity'] = row[col]
                    elif 'effort' in col_lower:
                        participant_ratings['effort'] = row[col]
                    elif 'surprising' in col_lower:
                        participant_ratings['surprise'] = row[col]
                    elif 'fair' in col_lower:
                        participant_ratings['fairness'] = row[col]
                    elif 'coherent' in col_lower:
                        participant_ratings['coherence'] = row[col]
                    elif 'enjoyable' in col_lower:
                        participant_ratings['enjoyability'] = row[col]

                if not participant_ratings:
                    continue

                individual_data.append({
                    'participant_id': participant_id,
                    'story': story_name,
                    'model': model_key,
                    'is_artificial': is_artificial,
                    **participant_ratings,
                    'surprise_metric': surprise_metric,
                    'coherence_knowitall': coherence_knowitall,
                    'coherence_human': coherence_human,
                    'fairplay_human': fairplay_human,
                    'fairplay_llm': fairplay_llm
                })

        df_individual = pd.DataFrame(individual_data)

        print(f"Prepared {len(df_individual)} individual observations")
        print(f"From {df_individual['participant_id'].nunique()} unique participants")
        print(f"Across {df_individual['story'].nunique()} stories")

        # Results dictionary
        results = {
            'models': {},
            'comparisons': {},
            'diagnostics': {}
        }

        # =========================================================================
        # MODEL 1: Surprise Rating ~ Surprise Metric
        # =========================================================================

        print("\n" + "=" * 70)
        print("MODEL 1: Surprise Rating ~ Surprise Metric")
        print("=" * 70 + "\n")

        surprise_data = df_individual[['participant_id', 'story', 'surprise', 'surprise_metric']].dropna()

        if len(surprise_data) >= 30:
            # Fit mixed effects model
            model_surprise = smf.mixedlm(
                formula="surprise ~ surprise_metric",
                data=surprise_data,
                groups=surprise_data["participant_id"]
            )

            try:
                result_surprise = model_surprise.fit(method='lbfgs')

                print(result_surprise.summary())

                # Extract key statistics
                beta = result_surprise.params['surprise_metric']
                se = result_surprise.bse['surprise_metric']
                p_value = result_surprise.pvalues['surprise_metric']
                ci_lower, ci_upper = result_surprise.conf_int().loc['surprise_metric']

                # Calculate ICC
                random_var = result_surprise.cov_re.iloc[0, 0]
                residual_var = result_surprise.scale
                icc = random_var / (random_var + residual_var)

                print("\n" + "-" * 70)
                print("KEY RESULTS:")
                print("-" * 70)
                print(f"Fixed effect of surprise_metric:")
                print(f"  β = {beta:.3f} (SE = {se:.3f})")
                print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"  p-value: {p_value:.6f}")
                print(f"\nIntraclass Correlation (ICC): {icc:.3f}")
                print(f"  → {icc * 100:.1f}% of variance is due to participant differences")
                print(f"\nSample size:")
                print(f"  {len(surprise_data)} observations")
                print(f"  {surprise_data['participant_id'].nunique()} participants")
                print(f"  {surprise_data['story'].nunique()} stories")

                # Store results
                results['models']['surprise'] = {
                    'fitted_model': result_surprise,
                    'beta': beta,
                    'se': se,
                    'p_value': p_value,
                    'ci': (ci_lower, ci_upper),
                    'icc': icc,
                    'n_obs': len(surprise_data),
                    'n_participants': surprise_data['participant_id'].nunique(),
                    'converged': True
                }

                # Also compute simple correlation for comparison
                corr_simple, p_simple = stats.spearmanr(surprise_data['surprise'],
                                                        surprise_data['surprise_metric'])

                results['comparisons']['surprise'] = {
                    'mixed_effects_beta': beta,
                    'mixed_effects_p': p_value,
                    'simple_correlation': corr_simple,
                    'simple_p': p_simple
                }

                print("\n" + "-" * 70)
                print("COMPARISON WITH SIMPLE CORRELATION:")
                print("-" * 70)
                print(f"Mixed effects β: {beta:.3f} (p = {p_value:.6f})")
                print(f"Simple correlation r: {corr_simple:.3f} (p = {p_simple:.6f})")
                print("\nNote: β and r are not directly comparable, but both measure")
                print("the strength of the relationship. Mixed effects accounts for")
                print("clustering and provides more accurate p-values.")

            except Exception as e:
                print(f"\n⚠ Warning: Model failed to converge")
                print(f"Error details: {str(e)}")
                print("\nTroubleshooting suggestions:")
                print("  1. Check if you have enough data (need n ≥ 30)")
                print("  2. Try scaling your variables")
                print("  3. Check for outliers or missing values")
                print("  4. Ensure participant_id is properly formatted")
                results['models']['surprise'] = {
                    'converged': False,
                    'error': str(e),
                    'n_obs': len(surprise_data)
                }
        else:
            print(f"⚠ Insufficient data for mixed effects model")
            print(f"  Current: {len(surprise_data)} observations")
            print(f"  Required: At least 30 observations")
            print(f"  Participants: {surprise_data['participant_id'].nunique() if len(surprise_data) > 0 else 0}")
            results['models']['surprise'] = {
                'converged': False,
                'error': 'Insufficient data',
                'n_obs': len(surprise_data)
            }

        # =========================================================================
        # MODEL 2: Fairness Rating ~ Fairplay (Human-based)
        # =========================================================================

        print("\n" + "=" * 70)
        print("MODEL 2: Fairness Rating ~ Fairplay (Human-based)")
        print("=" * 70 + "\n")

        fairness_data = df_individual[['participant_id', 'story', 'fairness', 'fairplay_human']].dropna()

        if len(fairness_data) >= 30:
            model_fairness = smf.mixedlm(
                formula="fairness ~ fairplay_human",
                data=fairness_data,
                groups=fairness_data["participant_id"]
            )

            try:
                result_fairness = model_fairness.fit(method='lbfgs')

                print(result_fairness.summary())

                beta = result_fairness.params['fairplay_human']
                se = result_fairness.bse['fairplay_human']
                p_value = result_fairness.pvalues['fairplay_human']

                random_var = result_fairness.cov_re.iloc[0, 0]
                residual_var = result_fairness.scale
                icc = random_var / (random_var + residual_var)
                ci_lower, ci_upper = result_fairness.conf_int().loc['fairplay_human']

                print("\n" + "-" * 70)
                print("KEY RESULTS:")
                print(f"β = {beta:.3f} (SE = {se:.3f}), p = {p_value:.6f}")
                print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"ICC = {icc:.3f}")
                print(
                    f"\nSample: {len(fairness_data)} observations from {fairness_data['participant_id'].nunique()} participants")

                results['models']['fairness_human'] = {
                    'fitted_model': result_fairness,
                    'beta': beta,
                    'se': se,
                    'p_value': p_value,
                    'ci': (ci_lower, ci_upper),
                    'icc': icc,
                    'n_obs': len(fairness_data),
                    'n_participants': fairness_data['participant_id'].nunique(),
                    'converged': True
                }

            except Exception as e:
                print(f"\n⚠ Warning: Model failed to converge")
                print(f"Error: {str(e)}")
                results['models']['fairness_human'] = {
                    'converged': False,
                    'error': str(e),
                    'n_obs': len(fairness_data)
                }
        else:
            print(f"⚠ Insufficient data for mixed effects model")
            print(f"  Current: {len(fairness_data)} observations")
            print(f"  Required: At least 30 observations")
            results['models']['fairness_human'] = {
                'converged': False,
                'error': 'Insufficient data',
                'n_obs': len(fairness_data)
            }

        # =========================================================================
        # MODEL 3: Coherence Rating ~ Coherence (Human-based)
        # =========================================================================

        print("\n" + "=" * 70)
        print("MODEL 3: Coherence Rating ~ Coherence (Human-based)")
        print("=" * 70 + "\n")

        coherence_human_data = df_individual[['participant_id', 'story', 'coherence', 'coherence_human']].dropna()

        if len(coherence_human_data) >= 30:
            model_coherence_human = smf.mixedlm(
                formula="coherence ~ coherence_human",
                data=coherence_human_data,
                groups=coherence_human_data["participant_id"]
            )

            try:
                result_coherence_human = model_coherence_human.fit(method='lbfgs')

                print(result_coherence_human.summary())

                beta = result_coherence_human.params['coherence_human']
                se = result_coherence_human.bse['coherence_human']
                p_value = result_coherence_human.pvalues['coherence_human']
                ci_lower, ci_upper = result_coherence_human.conf_int().loc['coherence_human']

                random_var = result_coherence_human.cov_re.iloc[0, 0]
                residual_var = result_coherence_human.scale
                icc = random_var / (random_var + residual_var)

                print("\n" + "-" * 70)
                print("KEY RESULTS:")
                print(f"β = {beta:.3f} (SE = {se:.3f}), p = {p_value:.6f}")
                print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"ICC = {icc:.3f}")
                print(
                    f"Sample: {len(coherence_human_data)} observations from {coherence_human_data['participant_id'].nunique()} participants")

                results['models']['coherence_human'] = {
                    'fitted_model': result_coherence_human,
                    'beta': beta,
                    'se': se,
                    'p_value': p_value,
                    'ci': (ci_lower, ci_upper),
                    'icc': icc,
                    'n_obs': len(coherence_human_data),
                    'n_participants': coherence_human_data['participant_id'].nunique(),
                    'converged': True
                }

            except Exception as e:
                print(f"\n⚠ Warning: Model failed to converge")
                print(f"Error: {str(e)}")
                results['models']['coherence_human'] = {
                    'converged': False,
                    'error': str(e),
                    'n_obs': len(coherence_human_data)
                }
        else:
            print(f"⚠ Insufficient data for mixed effects model")
            print(f"  Current: {len(coherence_human_data)} observations")
            results['models']['coherence_human'] = {
                'converged': False,
                'error': 'Insufficient data',
                'n_obs': len(coherence_human_data)
            }

        # =========================================================================
        # MODEL 4: Coherence Rating ~ Coherence (LLM-based / Know-it-all)
        # =========================================================================

        print("\n" + "=" * 70)
        print("MODEL 4: Coherence Rating ~ Coherence (LLM-based)")
        print("=" * 70 + "\n")

        coherence_llm_data = df_individual[['participant_id', 'story', 'coherence', 'coherence_knowitall']].dropna()

        if len(coherence_llm_data) >= 30:
            model_coherence_llm = smf.mixedlm(
                formula="coherence ~ coherence_knowitall",
                data=coherence_llm_data,
                groups=coherence_llm_data["participant_id"]
            )

            try:
                result_coherence_llm = model_coherence_llm.fit(method='lbfgs')

                print(result_coherence_llm.summary())

                beta = result_coherence_llm.params['coherence_knowitall']
                se = result_coherence_llm.bse['coherence_knowitall']
                p_value = result_coherence_llm.pvalues['coherence_knowitall']
                ci_lower, ci_upper = result_coherence_llm.conf_int().loc['coherence_knowitall']

                random_var = result_coherence_llm.cov_re.iloc[0, 0]
                residual_var = result_coherence_llm.scale
                icc = random_var / (random_var + residual_var)

                print("\n" + "-" * 70)
                print("KEY RESULTS:")
                print(f"β = {beta:.3f} (SE = {se:.3f}), p = {p_value:.6f}")
                print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"ICC = {icc:.3f}")
                print(
                    f"Sample: {len(coherence_llm_data)} observations from {coherence_llm_data['participant_id'].nunique()} participants")

                results['models']['coherence_knowitall'] = {
                    'fitted_model': result_coherence_llm,
                    'beta': beta,
                    'se': se,
                    'p_value': p_value,
                    'ci': (ci_lower, ci_upper),
                    'icc': icc,
                    'n_obs': len(coherence_llm_data),
                    'n_participants': coherence_llm_data['participant_id'].nunique(),
                    'converged': True
                }

            except Exception as e:
                print(f"\n⚠ Warning: Model failed to converge")
                print(f"Error: {str(e)}")
                results['models']['coherence_knowitall'] = {
                    'converged': False,
                    'error': str(e),
                    'n_obs': len(coherence_llm_data)
                }
        else:
            print(f"⚠ Insufficient data for mixed effects model")
            print(f"  Current: {len(coherence_llm_data)} observations")
            results['models']['coherence_knowitall'] = {
                'converged': False,
                'error': 'Insufficient data',
                'n_obs': len(coherence_llm_data)
            }

        # =========================================================================
        # MODEL 5: Fairness Rating ~ Fairplay (LLM-based)
        # =========================================================================

        print("\n" + "=" * 70)
        print("MODEL 5: Fairness Rating ~ Fairplay (LLM-based)")
        print("=" * 70 + "\n")

        fairness_llm_data = df_individual[['participant_id', 'story', 'fairness', 'fairplay_llm']].dropna()

        if len(fairness_llm_data) >= 30:
            model_fairness_llm = smf.mixedlm(
                formula="fairness ~ fairplay_llm",
                data=fairness_llm_data,
                groups=fairness_llm_data["participant_id"]
            )

            try:
                result_fairness_llm = model_fairness_llm.fit(method='lbfgs')

                print(result_fairness_llm.summary())

                beta = result_fairness_llm.params['fairplay_llm']
                se = result_fairness_llm.bse['fairplay_llm']
                p_value = result_fairness_llm.pvalues['fairplay_llm']
                ci_lower, ci_upper = result_fairness_llm.conf_int().loc['fairplay_llm']

                random_var = result_fairness_llm.cov_re.iloc[0, 0]
                residual_var = result_fairness_llm.scale
                icc = random_var / (random_var + residual_var)

                print(f"\nKey results:")
                print(f"β = {beta:.3f} (SE = {se:.3f}), p = {p_value:.6f}")
                print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"ICC = {icc:.3f}")
                print(
                    f"Sample: {len(fairness_llm_data)} observations from {fairness_llm_data['participant_id'].nunique()} participants")

                results['models']['fairness_llm'] = {
                    'fitted_model': result_fairness_llm,
                    'beta': beta,
                    'se': se,
                    'p_value': p_value,
                    'ci': (ci_lower, ci_upper),
                    'icc': icc,
                    'n_obs': len(fairness_llm_data),
                    'n_participants': fairness_llm_data['participant_id'].nunique(),
                    'converged': True
                }

            except Exception as e:
                print(f"\n⚠ Warning: Model failed to converge")
                print(f"Error: {str(e)}")
                results['models']['fairness_llm'] = {
                    'converged': False,
                    'error': str(e),
                    'n_obs': len(fairness_llm_data)
                }
        else:
            print(f"⚠ Insufficient data for mixed effects model")
            print(f"  Current: {len(fairness_llm_data)} observations")
            results['models']['fairness_llm'] = {
                'converged': False,
                'error': 'Insufficient data',
                'n_obs': len(fairness_llm_data)
            }

        # =========================================================================
        # CROSS-METRIC MODELS: Fairness Rating ~ Coherence Metrics
        # =========================================================================

        print("\n" + "=" * 70)
        print("CROSS-METRIC ANALYSIS: Does Coherence Predict Fairness Ratings?")
        print("=" * 70 + "\n")

        # MODEL 6: Fairness ~ Coherence (Human-based)
        print("\n" + "-" * 70)
        print("MODEL 6: Fairness Rating ~ Coherence (Human-based)")
        print("-" * 70 + "\n")

        fairness_coherence_human_data = df_individual[
            ['participant_id', 'story', 'fairness', 'coherence_human']].dropna()

        if len(fairness_coherence_human_data) >= 30:
            model_fair_coh_human = smf.mixedlm(
                formula="fairness ~ coherence_human",
                data=fairness_coherence_human_data,
                groups=fairness_coherence_human_data["participant_id"]
            )

            try:
                result_fair_coh_human = model_fair_coh_human.fit(method='lbfgs')

                print(result_fair_coh_human.summary())

                beta = result_fair_coh_human.params['coherence_human']
                se = result_fair_coh_human.bse['coherence_human']
                p_value = result_fair_coh_human.pvalues['coherence_human']
                ci_lower, ci_upper = result_fair_coh_human.conf_int().loc['coherence_human']

                random_var = result_fair_coh_human.cov_re.iloc[0, 0]
                residual_var = result_fair_coh_human.scale
                icc = random_var / (random_var + residual_var)

                print("\n" + "-" * 70)
                print("KEY RESULTS:")
                print(f"β = {beta:.3f} (SE = {se:.3f}), p = {p_value:.6f}")
                print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"ICC = {icc:.3f}")
                print(f"Sample: {len(fairness_coherence_human_data)} observations")

                if p_value < 0.05:
                    print("\n✓ SIGNIFICANT: Human-based coherence predicts fairness ratings!")
                else:
                    print("\n✗ NOT SIGNIFICANT: No relationship between coherence and fairness.")

                results['models']['fairness_by_coherence_human'] = {
                    'fitted_model': result_fair_coh_human,
                    'beta': beta,
                    'se': se,
                    'p_value': p_value,
                    'ci': (ci_lower, ci_upper),
                    'icc': icc,
                    'n_obs': len(fairness_coherence_human_data),
                    'n_participants': fairness_coherence_human_data['participant_id'].nunique(),
                    'converged': True
                }

            except Exception as e:
                print(f"\n⚠ Warning: Model failed to converge")
                print(f"Error: {str(e)}")
                results['models']['fairness_by_coherence_human'] = {
                    'converged': False,
                    'error': str(e),
                    'n_obs': len(fairness_coherence_human_data)
                }
        else:
            print(f"⚠ Insufficient data (n={len(fairness_coherence_human_data)})")
            results['models']['fairness_by_coherence_human'] = {
                'converged': False,
                'error': 'Insufficient data',
                'n_obs': len(fairness_coherence_human_data)
            }

        # MODEL 7: Fairness ~ Coherence (LLM-based)
        print("\n" + "-" * 70)
        print("MODEL 7: Fairness Rating ~ Coherence (LLM-based)")
        print("-" * 70 + "\n")

        fairness_coherence_llm_data = df_individual[
            ['participant_id', 'story', 'fairness', 'coherence_knowitall']].dropna()

        if len(fairness_coherence_llm_data) >= 30:
            model_fair_coh_llm = smf.mixedlm(
                formula="fairness ~ coherence_knowitall",
                data=fairness_coherence_llm_data,
                groups=fairness_coherence_llm_data["participant_id"]
            )

            try:
                result_fair_coh_llm = model_fair_coh_llm.fit(method='lbfgs')

                print(result_fair_coh_llm.summary())

                beta = result_fair_coh_llm.params['coherence_knowitall']
                se = result_fair_coh_llm.bse['coherence_knowitall']
                p_value = result_fair_coh_llm.pvalues['coherence_knowitall']
                ci_lower, ci_upper = result_fair_coh_llm.conf_int().loc['coherence_knowitall']

                random_var = result_fair_coh_llm.cov_re.iloc[0, 0]
                residual_var = result_fair_coh_llm.scale
                icc = random_var / (random_var + residual_var)

                print("\n" + "-" * 70)
                print("KEY RESULTS:")
                print(f"β = {beta:.3f} (SE = {se:.3f}), p = {p_value:.6f}")
                print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"ICC = {icc:.3f}")
                print(f"Sample: {len(fairness_coherence_llm_data)} observations")

                if p_value < 0.05:
                    print("\n✓ SIGNIFICANT: LLM-based coherence predicts fairness ratings!")
                else:
                    print("\n✗ NOT SIGNIFICANT: No relationship between coherence and fairness.")

                results['models']['fairness_by_coherence_llm'] = {
                    'fitted_model': result_fair_coh_llm,
                    'beta': beta,
                    'se': se,
                    'p_value': p_value,
                    'ci': (ci_lower, ci_upper),
                    'icc': icc,
                    'n_obs': len(fairness_coherence_llm_data),
                    'n_participants': fairness_coherence_llm_data['participant_id'].nunique(),
                    'converged': True
                }

            except Exception as e:
                print(f"\n⚠ Warning: Model failed to converge")
                print(f"Error: {str(e)}")
                results['models']['fairness_by_coherence_llm'] = {
                    'converged': False,
                    'error': str(e),
                    'n_obs': len(fairness_coherence_llm_data)
                }
        else:
            print(f"⚠ Insufficient data (n={len(fairness_coherence_llm_data)})")
            results['models']['fairness_by_coherence_llm'] = {
                'converged': False,
                'error': 'Insufficient data',
                'n_obs': len(fairness_coherence_llm_data)
            }

        # =========================================================================
        # DIAGNOSTIC: Check ICC values
        # =========================================================================

        print("\n" + "=" * 70)
        print("DIAGNOSTIC: Intraclass Correlation Coefficients")
        print("=" * 70 + "\n")
        print("ICC tells you how much variance is due to participant differences.")
        print("Higher ICC → more important to use mixed effects models.\n")

        for model_name, model_results in results['models'].items():
            if model_results.get('converged') and 'icc' in model_results:
                icc = model_results['icc']
                print(f"{model_name:20s}: ICC = {icc:.3f} ({icc * 100:.1f}% of variance)")

                if icc > 0.3:
                    print(f"  → High clustering! Mixed effects model is essential.")
                elif icc > 0.1:
                    print(f"  → Moderate clustering. Mixed effects model is recommended.")
                else:
                    print(f"  → Low clustering. Mixed effects provides modest improvement.")

        # =========================================================================
        # SUMMARY TABLE
        # =========================================================================

        print("\n" + "=" * 70)
        print("SUMMARY: All Models")
        print("=" * 70 + "\n")

        # Group models by category
        rating_models = {
            'Surprise Models': ['surprise'],
            'Fairness Models': ['fairness_human', 'fairness_llm'],
            'Coherence Models': ['coherence_human', 'coherence_knowitall'],
            'Cross-Metric (Fairness by Coherence)': ['fairness_by_coherence_human', 'fairness_by_coherence_llm']
        }

        for category, model_names in rating_models.items():
            summary_data = []

            for model_name in model_names:
                if model_name in results['models']:
                    model_results = results['models'][model_name]
                    if model_results.get('converged'):
                        # Only add to summary if all required keys exist
                        if all(key in model_results for key in ['beta', 'se', 'p_value', 'n_obs', 'icc']):
                            summary_data.append({
                                'Model': model_name,
                                'β': f"{model_results['beta']:.3f}",
                                'SE': f"{model_results['se']:.3f}",
                                'p-value': f"{model_results['p_value']:.6f}",
                                'Sig': '***' if model_results['p_value'] < 0.001 else
                                '**' if model_results['p_value'] < 0.01 else
                                '*' if model_results['p_value'] < 0.05 else 'ns',
                                'n': model_results['n_obs'],
                                'ICC': f"{model_results['icc']:.3f}"
                            })

            if summary_data:
                print(f"\n{category}:")
                print("-" * 70)
                summary_df = pd.DataFrame(summary_data)
                print(summary_df.to_string(index=False))

        if not any(results['models'].get(m, {}).get('converged') for m in
                   ['surprise', 'fairness_human', 'fairness_llm', 'coherence_human',
                    'coherence_knowitall', 'fairness_by_coherence_human', 'fairness_by_coherence_llm']):
            print("\nNo models converged successfully with complete results.")

        # Show which models failed and why
        print("\n" + "-" * 70)
        print("MODEL STATUS:")
        print("-" * 70)
        for model_name, model_results in results['models'].items():
            if model_results.get('converged'):
                print(f"✓ {model_name:25s} - SUCCESS")
            else:
                error_msg = model_results.get('error', 'Unknown error')
                n_obs = model_results.get('n_obs', 0)
                print(f"✗ {model_name:25s} - FAILED: {error_msg} (n={n_obs})")
        print("=" * 70)

        # =========================================================================
        # INTERPRETATION: Cross-Metric Relationships
        # =========================================================================

        print("\n" + "=" * 70)
        print("INTERPRETATION: Cross-Metric Relationships")
        print("=" * 70 + "\n")

        print("Testing whether coherence metrics predict fairness ratings:")
        print("(This tells us if participants who rate stories as fair also rate them as coherent)\n")

        # Check fairness by coherence (human)
        if 'fairness_by_coherence_human' in results['models']:
            model = results['models']['fairness_by_coherence_human']
            if model.get('converged'):
                beta = model['beta']
                p = model['p_value']

                print(f"1. Human-based Coherence → Fairness Rating:")
                print(f"   β = {beta:.3f}, p = {p:.6f}")

                if p < 0.05:
                    direction = "increases" if beta > 0 else "decreases"
                    print(f"   ✓ SIGNIFICANT: Higher human coherence {direction} fairness ratings")
                    print(
                        f"   → Participants who see the story as coherent ALSO rate it as {'more' if beta > 0 else 'less'} fair")
                else:
                    print(f"   ✗ NOT SIGNIFICANT: Coherence and fairness are independent")
                print()

        # Check fairness by coherence (LLM)
        if 'fairness_by_coherence_llm' in results['models']:
            model = results['models']['fairness_by_coherence_llm']
            if model.get('converged'):
                beta = model['beta']
                p = model['p_value']

                print(f"2. LLM-based Coherence → Fairness Rating:")
                print(f"   β = {beta:.3f}, p = {p:.6f}")

                if p < 0.05:
                    direction = "increases" if beta > 0 else "decreases"
                    print(f"   ✓ SIGNIFICANT: Higher LLM coherence {direction} fairness ratings")
                    print(
                        f"   → Stories that are coherent to know-it-all readers are rated as {'more' if beta > 0 else 'less'} fair")
                else:
                    print(f"   ✗ NOT SIGNIFICANT: LLM coherence doesn't predict fairness")
                print()

        # Compare the two
        human_converged = results['models'].get('fairness_by_coherence_human', {}).get('converged', False)
        llm_converged = results['models'].get('fairness_by_coherence_llm', {}).get('converged', False)

        if human_converged and llm_converged:
            human_beta = results['models']['fairness_by_coherence_human']['beta']
            human_p = results['models']['fairness_by_coherence_human']['p_value']
            llm_beta = results['models']['fairness_by_coherence_llm']['beta']
            llm_p = results['models']['fairness_by_coherence_llm']['p_value']

            print("COMPARISON:")
            print(f"Human coherence → Fairness: β = {human_beta:.3f}, p = {human_p:.6f}")
            print(f"LLM coherence → Fairness:   β = {llm_beta:.3f}, p = {llm_p:.6f}")

            if human_p < 0.05 and llm_p < 0.05:
                print("\nBoth coherence metrics predict fairness ratings!")
                if abs(human_beta) > abs(llm_beta):
                    print("Human-based coherence has a stronger effect.")
                else:
                    print("LLM-based coherence has a stronger effect.")
            elif human_p < 0.05:
                print("\nOnly human-based coherence predicts fairness (not LLM-based)")
            elif llm_p < 0.05:
                print("\nOnly LLM-based coherence predicts fairness (not human-based)")
            else:
                print("\nNeither coherence metric significantly predicts fairness ratings")

        print("=" * 70)

        return results, df_individual


    def calculate_curve_differences(self):
        """Calculate differences between reading curves."""
        results = []

        for story_name in self.human_data.keys():
            model_key, story_id = story_name.split()
            is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

            human_dist, _, _ = self.get_human_distribution(story_name)
            if human_dist is None:
                continue

            model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
            if is_artificial:
                sampling_dist, naive_dist, true_idx, sampling_ci, nsamp = self.get_llm_distributions(
                    model_key, model_key2, story_id
                )
            else:
                sampling_dist, naive_dist, true_idx, sampling_ci, nsamp = self.get_llm_distributions(
                    "gemini-1.5-flash", model_key, story_id
                )

            if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                continue

            human_true = human_dist[:, true_idx]
            naive_true = naive_dist[:, true_idx]

            human_naive_diff = float(np.trapz(human_true - naive_true) / len(human_true))

            threshold = 1 / len(human_true)
            human_naive_exceeds = np.sum(human_true - naive_true > threshold)

            result = {
                'story': story_name,
                'model': model_key,
                'is_artificial': is_artificial,
                'human_naive_diff': human_naive_diff,
                'human_naive_exceeds': human_naive_exceeds,
                'n_paragraphs': len(human_true)
            }

            if is_artificial and sampling_dist is not None and len(sampling_dist) == len(human_dist):
                sampling_true = sampling_dist[:, true_idx]
                knowitall_naive_diff = float(np.trapz(sampling_true - naive_true) / len(sampling_true))
                knowitall_naive_exceeds = np.sum(sampling_true - naive_true > threshold)

                result['knowitall_naive_diff'] = knowitall_naive_diff
                result['knowitall_naive_exceeds'] = knowitall_naive_exceeds

            results.append(result)

        return pd.DataFrame(results)


    def analyze_reader_performance(self):
        """Analyze individual reader performance."""
        reader_stats = []

        for story_name, data in self.human_data.items():
            df = data['df']
            response_cols = data['response_cols']

            final_responses = df[response_cols[-1]].dropna()
            if len(final_responses) == 0:
                continue

            true_culprit = final_responses.mode()[0] if len(final_responses.mode()) > 0 else None
            if true_culprit is None:
                continue

            for idx, row in df.iterrows():
                user_id = row[df.columns[1]]
                timestamp = row[df.columns[0]]

                correct_count = 0
                total_count = 0

                for col in response_cols:
                    if pd.notna(row[col]):
                        total_count += 1
                        if row[col] == true_culprit:
                            correct_count += 1

                if total_count > 0:
                    accuracy = correct_count / total_count

                    reader_stats.append({
                        'user_id': user_id,
                        'story': story_name,
                        'timestamp': timestamp,
                        'accuracy': accuracy,
                        'correct_count': correct_count,
                        'total_count': total_count,
                        'is_artificial': 'sherlock' not in story_name and "poirot" not in story_name
                    })

        df_stats = pd.DataFrame(reader_stats)

        if not df_stats.empty:
            story_avgs = df_stats.groupby('story')['accuracy'].transform('mean')
            df_stats['relative_performance'] = df_stats['accuracy'] - story_avgs

        return df_stats


    def analyze_subjective_correlations(self, skip_non_converged=True, naive_mode="default", exp_surprise_setting=0, run_mixed_effects=True):
        """
        Comprehensive correlation analysis between subjective ratings and metrics.

        Analyzes:
        1. Within subjective ratings:
           - Surprise vs. Coherence
           - Coherence & Surprise vs. Fairness
           - Surprise, Coherence & Fairness vs. Enjoyability

        2. Subjective vs. Quantitative:
           - Subjective Surprise vs. Measured Surprise (1 - naive_true)
           - Subjective Coherence vs. Measured Coherence (human_true or sampling_true)
           - Subjective Coherence vs. Measured Fair Play
           - Subjective Fairness vs. Measured Fair Play
           - Enjoyability vs. Coherence and Fair Play
        """

        print("\n" + "=" * 80)
        print("COMPREHENSIVE CORRELATION ANALYSIS")
        print("=" * 80 + "\n")

        # Naive distributions are computed with the specified naive_mode for this analysis
        saved_mode = self.naive_reader_mode
        self.set_naive_reader_mode(naive_mode)

        # Collect individual-level data with CORRECT participant IDs
        individual_data = []

        for story_name, data in self.human_data.items():
            df = data['df']
            ratings = data['ratings']

            if len(ratings) == 0:
                continue

            # Get computational measures
            human_dist, _, _ = self.get_human_distribution(story_name)
            if human_dist is None:
                continue

            # measured_surprise uses default naive (raw LLM distribution)
            self.set_naive_reader_mode("default")
            naive_dist, true_idx = self.get_naive_reader_probs(story_name)
            self.set_naive_reader_mode(naive_mode)
            if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                continue

            # Super_naive distributions — always computed with super_naive_report
            self.set_naive_reader_mode("super_naive_report")
            sn_dist, _ = self.get_naive_reader_probs(story_name)
            self.set_naive_reader_mode(naive_mode)
            sn_valid = sn_dist is not None and len(sn_dist) == len(human_dist)
            measured_surprise_sn = (
                1 - np.mean(sn_dist[:, true_idx]) if sn_valid else None
            )

            model_key, story_id = story_name.split()
            is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

            # Get sampling distribution
            if not is_artificial:
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    REAL_STORY_S_MODEL, model_key, story_id
                )
            else:
                model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    model_key, model_key2, story_id
                )

            if sampling_dist is None or len(sampling_dist) != len(human_dist):
                continue

            # Calculate computational measures
            human_true = human_dist[:, true_idx]
            naive_true = naive_dist[:, true_idx]
            sampling_true = sampling_dist[:, true_idx]

            measured_surprise = 1 - np.mean(naive_true)  # Low naive prob = high surprise
            measured_coherence_human = np.mean(human_true)
            measured_coherence_llm = np.mean(sampling_true)
            measured_fairplay_human = np.mean(human_true - naive_true)
            measured_fairplay_llm = np.mean(sampling_true - naive_true)

            # Super-naive versions of FP (replace naive baseline with step-function)
            sn_true = sn_dist[:, true_idx] if sn_valid else None
            measured_fairplay_human_sn = float(np.mean(human_true - sn_true)) if sn_valid else None
            measured_fairplay_llm_sn   = float(np.mean(sampling_true - sn_true)) if sn_valid else None

            # Human-guessing surprise: use human guesses as the "naive" baseline
            measured_surprise_human = 1 - np.mean(human_true)

            # Experienced reader 0-shot: surprise and fair play
            measured_surprise_exp = None
            measured_fairplay_exp = None
            measured_fairplay_exp_sn = None
            llm_key, _ = self._resolve_llm_key(story_name)
            exp_dist, exp_idx = self.get_experienced_reader_distributions(
                *llm_key, setting=exp_surprise_setting
            )
            measured_coherence_exp = None
            if exp_dist is not None and exp_idx is not None and len(exp_dist) == len(human_dist):
                exp_true = exp_dist[:, exp_idx]
                measured_surprise_exp = 1 - np.mean(exp_true)
                measured_coherence_exp = float(np.mean(exp_true))
                measured_fairplay_exp = float(np.mean(exp_true - naive_true))
                if sn_valid:
                    measured_fairplay_exp_sn = float(np.mean(exp_true - sn_dist[:, true_idx]))

            # Extract rating columns
            rating_cols = df.columns[-6:]

            # Loop through each participant with CORRECT ID
            for idx, row in df.iterrows():
                # FIX: Use actual user ID from column 1, not row index!
                participant_id = row[df.columns[1]]

                participant_ratings = {}

                for col in rating_cols:
                    col_lower = col.lower()
                    if 'familiarity' in col_lower:
                        participant_ratings['familiarity'] = row[col]
                    elif 'effort' in col_lower:
                        participant_ratings['effort'] = row[col]
                    elif 'surprising' in col_lower:
                        participant_ratings['surprise'] = row[col]
                    elif 'fair' in col_lower:
                        participant_ratings['fairness'] = row[col]
                    elif 'coherent' in col_lower:
                        participant_ratings['coherence'] = row[col]
                    elif 'enjoyable' in col_lower:
                        participant_ratings['enjoyability'] = row[col]

                if not participant_ratings:
                    continue

                individual_data.append({
                    'participant_id': participant_id,
                    'story': story_name,
                    'model': model_key,
                    'is_artificial': is_artificial,
                    **participant_ratings,
                    'measured_surprise': measured_surprise,
                    'measured_surprise_human': measured_surprise_human,
                    'measured_surprise_exp': measured_surprise_exp,
                    'measured_surprise_sn': measured_surprise_sn,
                    'measured_coherence_human': measured_coherence_human,
                    'measured_coherence_exp': measured_coherence_exp,
                    'measured_coherence_llm': measured_coherence_llm,
                    'measured_fairplay_human': measured_fairplay_human,
                    'measured_fairplay_human_sn': measured_fairplay_human_sn,
                    'measured_fairplay_exp': measured_fairplay_exp,
                    'measured_fairplay_exp_sn': measured_fairplay_exp_sn,
                    'measured_fairplay_llm': measured_fairplay_llm,
                    'measured_fairplay_llm_sn': measured_fairplay_llm_sn,
                })

        df_data = pd.DataFrame(individual_data)

        print(f"Prepared {len(df_data)} individual observations")
        print(f"From {df_data['participant_id'].nunique()} unique participants")
        print(f"Across {df_data['story'].nunique()} stories\n")

        results = {}

        # =====================================================================
        # PART 1: CORRELATIONS WITHIN SUBJECTIVE RATINGS
        # =====================================================================

        print("=" * 80)
        print("PART 1: CORRELATIONS WITHIN SUBJECTIVE RATINGS")
        print("=" * 80 + "\n")

        subjective_pairs = [
            ('surprise', 'coherence', 'Surprise vs. Coherence'),
            ('surprise', 'fairness', 'Surprise vs. Fairness'),
            ('coherence', 'fairness', 'Coherence vs. Fairness'),
            ('surprise', 'enjoyability', 'Surprise vs. Enjoyability'),
            ('coherence', 'enjoyability', 'Coherence vs. Enjoyability'),
            ('fairness', 'enjoyability', 'Fairness vs. Enjoyability'),
        ]

        def _run_subjective_pairs(df_subset, result_key, header):
            print(f"\n{header}")
            print("-" * len(header))
            results['subjective_within'][result_key] = {}
            for var1, var2, label in subjective_pairs:
                print(f"{label}:")
                data_subset = df_subset[[var1, var2]].dropna()
                if len(data_subset) >= 30:
                    corr, p_value = stats.spearmanr(data_subset[var1], data_subset[var2])
                    results['subjective_within'][result_key][label] = {
                        'correlation': corr,
                        'p_value': p_value,
                        'n': len(data_subset)
                    }
                    print(f"  Spearman r = {corr:.3f}, p = {p_value:.4f}, n = {len(data_subset)}")
                else:
                    print(f"  Insufficient data (n = {len(data_subset)})")
                print()

        results['subjective_within'] = {}

        _run_subjective_pairs(df_data, 'all', 'All stories')
        _run_subjective_pairs(df_data[df_data['is_artificial']], 'artificial', 'Artificial stories only')
        _run_subjective_pairs(df_data[~df_data['is_artificial']], 'real', 'Real stories only (Poirot/Sherlock)')

        # =====================================================================
        # PART 2: SUBJECTIVE VS. QUANTITATIVE CORRELATIONS
        # =====================================================================

        print("=" * 80)
        print("PART 2: SUBJECTIVE VS. QUANTITATIVE CORRELATIONS")
        print("=" * 80 + "\n")

        exp_short = EXPERIENCED_READER_SHORT_LABELS.get(exp_surprise_setting, str(exp_surprise_setting))
        quantitative_pairs = [
            ('surprise', 'measured_surprise',    'Subjective Surprise vs. Measured Surprise (default naive)'),
            ('surprise', 'measured_surprise_sn', 'Subjective Surprise vs. Measured Surprise (super-naive)'),
            ('surprise', 'measured_surprise_human', 'Subjective Surprise vs. Human-Guessing Surprise'),
            ('surprise', 'measured_surprise_exp', f'Subjective Surprise vs. Exp Reader Surprise ({exp_short})'),
            ('coherence', 'measured_coherence_human', 'Subjective Coherence vs. Measured Coherence (Human)'),
            ('coherence', 'measured_coherence_exp',   'Subjective Coherence vs. Measured Coherence (Exp)'),
            ('coherence', 'measured_coherence_llm',   'Subjective Coherence vs. Measured Coherence (LLM)'),
            ('coherence', 'measured_fairplay_human',    'Subjective Coherence vs. Measured Fair Play (Human, default)'),
            ('coherence', 'measured_fairplay_human_sn', 'Subjective Coherence vs. Measured Fair Play (Human, super-naive)'),
            ('coherence', 'measured_fairplay_exp',      'Subjective Coherence vs. Measured Fair Play (Exp, default)'),
            ('coherence', 'measured_fairplay_exp_sn',   'Subjective Coherence vs. Measured Fair Play (Exp, super-naive)'),
            ('coherence', 'measured_fairplay_llm',      'Subjective Coherence vs. Measured Fair Play (LLM, default)'),
            ('coherence', 'measured_fairplay_llm_sn',   'Subjective Coherence vs. Measured Fair Play (LLM, super-naive)'),
            ('fairness', 'measured_coherence_human', 'Subjective Fairness vs. Measured Coherence (Human)'),
            ('fairness', 'measured_coherence_exp',   'Subjective Fairness vs. Measured Coherence (Exp)'),
            ('fairness', 'measured_coherence_llm',   'Subjective Fairness vs. Measured Coherence (LLM)'),
            ('fairness', 'measured_fairplay_human',    'Subjective Fairness vs. Measured Fair Play (Human, default)'),
            ('fairness', 'measured_fairplay_human_sn', 'Subjective Fairness vs. Measured Fair Play (Human, super-naive)'),
            ('fairness', 'measured_fairplay_exp',      'Subjective Fairness vs. Measured Fair Play (Exp, default)'),
            ('fairness', 'measured_fairplay_exp_sn',   'Subjective Fairness vs. Measured Fair Play (Exp, super-naive)'),
            ('fairness', 'measured_fairplay_llm',      'Subjective Fairness vs. Measured Fair Play (LLM, default)'),
            ('fairness', 'measured_fairplay_llm_sn',   'Subjective Fairness vs. Measured Fair Play (LLM, super-naive)'),
            ('enjoyability', 'measured_coherence_human', 'Enjoyability vs. Measured Coherence (Human)'),
            ('enjoyability', 'measured_coherence_exp',   'Enjoyability vs. Measured Coherence (Exp)'),
            ('enjoyability', 'measured_coherence_llm',   'Enjoyability vs. Measured Coherence (LLM)'),
            ('enjoyability', 'measured_fairplay_human',    'Enjoyability vs. Measured Fair Play (Human, default)'),
            ('enjoyability', 'measured_fairplay_human_sn', 'Enjoyability vs. Measured Fair Play (Human, super-naive)'),
            ('enjoyability', 'measured_fairplay_exp',      'Enjoyability vs. Measured Fair Play (Exp, default)'),
            ('enjoyability', 'measured_fairplay_exp_sn',   'Enjoyability vs. Measured Fair Play (Exp, super-naive)'),
            ('enjoyability', 'measured_fairplay_llm',      'Enjoyability vs. Measured Fair Play (LLM, default)'),
            ('enjoyability', 'measured_fairplay_llm_sn',   'Enjoyability vs. Measured Fair Play (LLM, super-naive)'),
        ]

        results['subjective_vs_quantitative'] = {}

        for subj_var, quant_var, label in quantitative_pairs:
            print(f"{label}:")

            data_subset = df_data[[subj_var, quant_var]].dropna()

            if len(data_subset) >= 30:
                corr, p_value = stats.spearmanr(data_subset[subj_var], data_subset[quant_var])

                results['subjective_vs_quantitative'][label] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n': len(data_subset)
                }

                print(f"  Spearman r = {corr:.3f}, p = {p_value:.4f}, n = {len(data_subset)}")
            else:
                print(f"  Insufficient data (n = {len(data_subset)})")

            print()

        # =====================================================================
        # PART 2b: STORY-LEVEL CORRELATIONS (subjective vs quantitative)
        # =====================================================================

        print("=" * 80)
        print("PART 2b: STORY-LEVEL CORRELATIONS (averaging ratings per story first)")
        print("  This removes within-story noise from different annotators")
        print("=" * 80 + "\n")

        # Average subjective ratings per story
        subj_cols = ['surprise', 'coherence', 'fairness', 'enjoyability']
        quant_cols = ['measured_surprise', 'measured_surprise_sn', 'measured_surprise_human', 'measured_surprise_exp',
                      'measured_coherence_human', 'measured_coherence_exp', 'measured_coherence_llm',
                      'measured_fairplay_human', 'measured_fairplay_human_sn',
                      'measured_fairplay_exp', 'measured_fairplay_exp_sn',
                      'measured_fairplay_llm', 'measured_fairplay_llm_sn']
        story_level = df_data.groupby('story')[subj_cols + quant_cols].mean().dropna()

        results['story_level_correlations'] = {}

        for subj_var, quant_var, label in quantitative_pairs:
            if subj_var not in story_level.columns or quant_var not in story_level.columns:
                continue

            valid = story_level[[subj_var, quant_var]].dropna()

            if len(valid) >= 5:
                corr, p_value = stats.spearmanr(valid[subj_var], valid[quant_var])

                results['story_level_correlations'][label] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'n': len(valid)
                }

                print(f"{label}:")
                print(f"  Spearman r = {corr:.3f}, p = {p_value:.4f}, n = {len(valid)} stories")
                print()
            else:
                print(f"{label}: Insufficient stories (n = {len(valid)})")
                print()

        # =====================================================================
        # PART 2c: ORDINAL ANALYSIS (Kruskal-Wallis / trend test)
        # =====================================================================

        print("=" * 80)
        print("PART 2c: ORDINAL ANALYSIS")
        print("  Kruskal-Wallis test: does the computational metric differ across rating levels?")
        print("  This avoids Spearman's attenuation from ties on the 1-5 scale.")
        print("=" * 80 + "\n")

        results['ordinal_analysis'] = {}

        for subj_var, quant_var, label in quantitative_pairs:
            data_subset = df_data[[subj_var, quant_var]].dropna()

            if len(data_subset) < 30:
                continue

            # Group the continuous metric by rating level
            groups = []
            group_labels = []
            for level in sorted(data_subset[subj_var].unique()):
                group_vals = data_subset[data_subset[subj_var] == level][quant_var].values
                if len(group_vals) >= 2:
                    groups.append(group_vals)
                    group_labels.append(level)

            if len(groups) < 2:
                continue

            # Kruskal-Wallis: non-parametric test for differences across groups
            h_stat, kw_p = stats.kruskal(*groups)

            # Also compute group means to show the direction
            group_means = [(lbl, grp.mean(), len(grp)) for lbl, grp in zip(group_labels, groups)]

            # Jonckheere-Terpstra-like: test for monotonic trend using Spearman
            # between rating level and mean metric per level
            if len(group_means) >= 3:
                level_vals = [g[0] for g in group_means]
                mean_vals = [g[1] for g in group_means]
                trend_r, trend_p = stats.spearmanr(level_vals, mean_vals)
            else:
                trend_r, trend_p = np.nan, np.nan

            results['ordinal_analysis'][label] = {
                'h_stat': h_stat,
                'kw_p': kw_p,
                'trend_r': trend_r,
                'trend_p': trend_p,
                'group_means': group_means
            }

            sig = '***' if kw_p < 0.001 else '**' if kw_p < 0.01 else '*' if kw_p < 0.05 else 'ns'
            print(f"{label}:")
            print(f"  Kruskal-Wallis: H = {h_stat:.3f}, p = {kw_p:.4f} {sig}")
            if not np.isnan(trend_r):
                trend_sig = '***' if trend_p < 0.001 else '**' if trend_p < 0.01 else '*' if trend_p < 0.05 else 'ns'
                print(f"  Monotonic trend (across rating levels): r = {trend_r:.3f}, p = {trend_p:.4f} {trend_sig}")
            print(f"  Mean metric by rating level:")
            for lbl, mean, n in group_means:
                print(f"    Rating {lbl:.0f}: metric mean = {mean:.4f} (n={n})")
            print()

        # =====================================================================
        # PART 3: MIXED EFFECTS MODELS (with FIXED participant ID bug)
        # =====================================================================
        if not run_mixed_effects:
            print("[Mixed effects skipped — run_mixed_effects=False]\n")
            self.set_naive_reader_mode(saved_mode)
            return results, df_data

        print("=" * 80)
        print("PART 3: MIXED EFFECTS MODELS (Key Relationships)")
        print("=" * 80 + "\n")

        # Filter to participants with at least 3 observations
        participant_counts = df_data.groupby('participant_id').size()
        valid_participants = participant_counts[participant_counts >= 3].index
        df_filtered = df_data[df_data['participant_id'].isin(valid_participants)]

        print(f"FILTERING: Keeping participants with ≥3 observations")
        print(f"  Original: {df_data['participant_id'].nunique()} participants, {len(df_data)} observations")
        print(f"  Filtered: {df_filtered['participant_id'].nunique()} participants, {len(df_filtered)} observations")
        print(
            f"  Removed: {df_data['participant_id'].nunique() - df_filtered['participant_id'].nunique()} participants\n")

        results['mixed_models'] = {}

        if len(df_filtered) < 30 or df_filtered['participant_id'].nunique() < 5:
            print("  Insufficient data after filtering. Skipping mixed models.\n")
        else:
            # Model 1: Surprise rating ~ Measured surprise
            print("Model 1: Surprise Rating ~ Measured Surprise")
            print("-" * 80)

            surprise_data = df_filtered[['participant_id', 'surprise', 'measured_surprise']].dropna()

            if len(surprise_data) >= 30 and surprise_data['participant_id'].nunique() >= 5:
                try:
                    model = smf.mixedlm(
                        formula="surprise ~ measured_surprise",
                        data=surprise_data,
                        groups=surprise_data["participant_id"]
                    )
                    result = model.fit(method='lbfgs')

                    log_likelihood = result.llf
                    intercept_se = result.bse['Intercept']
                    not_converged = np.isinf(log_likelihood) or intercept_se > 1000

                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model shows signs of convergence failure!")
                            print(f"   Log-likelihood: {log_likelihood}")
                            print(f"   Intercept SE: {intercept_se}")
                            print("   Results may be unreliable.\n")

                    results['mixed_models']['surprise'] = {
                        'beta': result.params['measured_surprise'],
                        'se': result.bse['measured_surprise'],
                        'p_value': result.pvalues['measured_surprise'],
                        'n_obs': len(surprise_data),
                        'n_groups': surprise_data['participant_id'].nunique(),
                        'converged': result.converged,
                        'log_likelihood': log_likelihood
                    }
                except Exception as e:
                    print(f"  Model failed: {e}")
            else:
                print(
                    f"  Insufficient data (n={len(surprise_data)}, groups={surprise_data['participant_id'].nunique()})")

            print("\n")

            # Model 2: Coherence rating ~ Measured coherence (Human)
            print("Model 2: Coherence Rating ~ Measured Coherence (Human)")
            print("-" * 80)

            coherence_data = df_filtered[['participant_id', 'coherence', 'measured_coherence_human']].dropna()

            if len(coherence_data) >= 30 and coherence_data['participant_id'].nunique() >= 5:
                try:
                    model = smf.mixedlm(
                        formula="coherence ~ measured_coherence_human",
                        data=coherence_data,
                        groups=coherence_data["participant_id"]
                    )
                    result = model.fit(method='lbfgs')

                    log_likelihood = result.llf
                    intercept_se = result.bse['Intercept']
                    not_converged = np.isinf(log_likelihood) or intercept_se > 1000

                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model shows signs of convergence failure!")
                            print(f"   Log-likelihood: {log_likelihood}")
                            print(f"   Intercept SE: {intercept_se}")
                            print("   Results may be unreliable.\n")

                    results['mixed_models']['coherence_human'] = {
                        'beta': result.params['measured_coherence_human'],
                        'se': result.bse['measured_coherence_human'],
                        'p_value': result.pvalues['measured_coherence_human'],
                        'n_obs': len(coherence_data),
                        'n_groups': coherence_data['participant_id'].nunique(),
                        'converged': result.converged,
                        'log_likelihood': log_likelihood
                    }
                except Exception as e:
                    print(f"  Model failed: {e}")
            else:
                print(
                    f"  Insufficient data (n={len(coherence_data)}, groups={coherence_data['participant_id'].nunique()})")

            print("\n")

            # Model 3: Coherence rating ~ Measured coherence (LLM)
            print("Model 3: Coherence Rating ~ Measured Coherence (LLM)")
            print("-" * 80)

            coherence_llm_data = df_filtered[['participant_id', 'coherence', 'measured_coherence_llm']].dropna()

            if len(coherence_llm_data) >= 30 and coherence_llm_data['participant_id'].nunique() >= 5:
                try:
                    model = smf.mixedlm(
                        formula="coherence ~ measured_coherence_llm",
                        data=coherence_llm_data,
                        groups=coherence_llm_data["participant_id"]
                    )
                    result = model.fit(method='lbfgs')

                    log_likelihood = result.llf
                    intercept_se = result.bse['Intercept']
                    not_converged = np.isinf(log_likelihood) or intercept_se > 1000

                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model shows signs of convergence failure!")
                            print(f"   Log-likelihood: {log_likelihood}")
                            print(f"   Intercept SE: {intercept_se}")
                            print("   Results may be unreliable.\n")

                    results['mixed_models']['coherence_llm'] = {
                        'beta': result.params['measured_coherence_llm'],
                        'se': result.bse['measured_coherence_llm'],
                        'p_value': result.pvalues['measured_coherence_llm'],
                        'n_obs': len(coherence_llm_data),
                        'n_groups': coherence_llm_data['participant_id'].nunique(),
                        'converged': result.converged,
                        'log_likelihood': log_likelihood
                    }
                except Exception as e:
                    print(f"  Model failed: {e}")
            else:
                print(
                    f"  Insufficient data (n={len(coherence_llm_data)}, groups={coherence_llm_data['participant_id'].nunique()})")

            print("\n")

            # Model 4: Fairness rating ~ Measured fairplay (Human)
            print("Model 4: Fairness Rating ~ Measured Fair Play (Human)")
            print("-" * 80)

            fairness_data = df_filtered[['participant_id', 'fairness', 'measured_fairplay_human']].dropna()

            if len(fairness_data) >= 30 and fairness_data['participant_id'].nunique() >= 5:
                try:
                    model = smf.mixedlm(
                        formula="fairness ~ measured_fairplay_human",
                        data=fairness_data,
                        groups=fairness_data["participant_id"]
                    )
                    result = model.fit(method='lbfgs')

                    log_likelihood = result.llf
                    intercept_se = result.bse['Intercept']
                    not_converged = np.isinf(log_likelihood) or intercept_se > 1000

                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model shows signs of convergence failure!")
                            print(f"   Log-likelihood: {log_likelihood}")
                            print(f"   Intercept SE: {intercept_se}")
                            print("   Results may be unreliable.\n")

                    results['mixed_models']['fairness_human'] = {
                        'beta': result.params['measured_fairplay_human'],
                        'se': result.bse['measured_fairplay_human'],
                        'p_value': result.pvalues['measured_fairplay_human'],
                        'n_obs': len(fairness_data),
                        'n_groups': fairness_data['participant_id'].nunique(),
                        'converged': result.converged,
                        'log_likelihood': log_likelihood
                    }
                except Exception as e:
                    print(f"  Model failed: {e}")
            else:
                print(
                    f"  Insufficient data (n={len(fairness_data)}, groups={fairness_data['participant_id'].nunique()})")

            print("\n")

            # Model 5: Fairness rating ~ Measured fairplay (LLM)
            print("Model 5: Fairness Rating ~ Measured Fair Play (LLM)")
            print("-" * 80)

            fairness_llm_data = df_filtered[['participant_id', 'fairness', 'measured_fairplay_llm']].dropna()

            if len(fairness_llm_data) >= 30 and fairness_llm_data['participant_id'].nunique() >= 5:
                try:
                    model = smf.mixedlm(
                        formula="fairness ~ measured_fairplay_llm",
                        data=fairness_llm_data,
                        groups=fairness_llm_data["participant_id"]
                    )
                    result = model.fit(method='lbfgs')

                    log_likelihood = result.llf
                    intercept_se = result.bse['Intercept']
                    not_converged = np.isinf(log_likelihood) or intercept_se > 1000

                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model shows signs of convergence failure!")
                            print(f"   Log-likelihood: {log_likelihood}")
                            print(f"   Intercept SE: {intercept_se}")
                            print("   Results may be unreliable.\n")

                    results['mixed_models']['fairness_llm'] = {
                        'beta': result.params['measured_fairplay_llm'],
                        'se': result.bse['measured_fairplay_llm'],
                        'p_value': result.pvalues['measured_fairplay_llm'],
                        'n_obs': len(fairness_llm_data),
                        'n_groups': fairness_llm_data['participant_id'].nunique(),
                        'converged': result.converged,
                        'log_likelihood': log_likelihood
                    }
                except Exception as e:
                    print(f"  Model failed: {e}")
            else:
                print(
                    f"  Insufficient data (n={len(fairness_llm_data)}, groups={fairness_llm_data['participant_id'].nunique()})")

            print("\n")

            # Model 6: Enjoyability ~ Measured coherence (Human)
            print("Model 6: Enjoyability ~ Measured Coherence (Human)")
            print("-" * 80)

            enjoy_coh_data = df_filtered[['participant_id', 'enjoyability', 'measured_coherence_human']].dropna()

            if len(enjoy_coh_data) >= 30 and enjoy_coh_data['participant_id'].nunique() >= 5:
                try:
                    model = smf.mixedlm(
                        formula="enjoyability ~ measured_coherence_human",
                        data=enjoy_coh_data,
                        groups=enjoy_coh_data["participant_id"]
                    )
                    result = model.fit(method='lbfgs')

                    log_likelihood = result.llf
                    intercept_se = result.bse['Intercept']
                    not_converged = np.isinf(log_likelihood) or intercept_se > 1000

                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model shows signs of convergence failure!")
                            print(f"   Log-likelihood: {log_likelihood}")
                            print(f"   Intercept SE: {intercept_se}")
                            print("   Results may be unreliable.\n")

                    results['mixed_models']['enjoyability_coherence'] = {
                        'beta': result.params['measured_coherence_human'],
                        'se': result.bse['measured_coherence_human'],
                        'p_value': result.pvalues['measured_coherence_human'],
                        'n_obs': len(enjoy_coh_data),
                        'n_groups': enjoy_coh_data['participant_id'].nunique(),
                        'converged': result.converged,
                        'log_likelihood': log_likelihood
                    }
                except Exception as e:
                    print(f"  Model failed: {e}")
            else:
                print(
                    f"  Insufficient data (n={len(enjoy_coh_data)}, groups={enjoy_coh_data['participant_id'].nunique()})")

            print("\n")

            # Model 7: Enjoyability ~ Measured fairplay (Human)
            print("Model 7: Enjoyability ~ Measured Fair Play (Human)")
            print("-" * 80)

            enjoy_fp_data = df_filtered[['participant_id', 'enjoyability', 'measured_fairplay_human']].dropna()

            if len(enjoy_fp_data) >= 30 and enjoy_fp_data['participant_id'].nunique() >= 5:
                try:
                    model = smf.mixedlm(
                        formula="enjoyability ~ measured_fairplay_human",
                        data=enjoy_fp_data,
                        groups=enjoy_fp_data["participant_id"]
                    )
                    result = model.fit(method='lbfgs')

                    log_likelihood = result.llf
                    intercept_se = result.bse['Intercept']
                    not_converged = np.isinf(log_likelihood) or intercept_se > 1000

                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model shows signs of convergence failure!")
                            print(f"   Log-likelihood: {log_likelihood}")
                            print(f"   Intercept SE: {intercept_se}")
                            print("   Results may be unreliable.\n")

                    results['mixed_models']['enjoyability_fairplay'] = {
                        'beta': result.params['measured_fairplay_human'],
                        'se': result.bse['measured_fairplay_human'],
                        'p_value': result.pvalues['measured_fairplay_human'],
                        'n_obs': len(enjoy_fp_data),
                        'n_groups': enjoy_fp_data['participant_id'].nunique(),
                        'converged': result.converged,
                        'log_likelihood': log_likelihood
                    }
                except Exception as e:
                    print(f"  Model failed: {e}")
            else:
                print(
                    f"  Insufficient data (n={len(enjoy_fp_data)}, groups={enjoy_fp_data['participant_id'].nunique()})")

            print("\n")

        print("=" * 80)
        print("END OF CORRELATION ANALYSIS")
        print("=" * 80 + "\n")

        self.set_naive_reader_mode(saved_mode)
        return results, df_data

    def analyze_metric_correlations(self, skip_non_converged=True, min_nsamp=1, run_mixed_effects=True):
        """
        Correlation analysis between computed metrics (surprise, coherence, FP, FP>1) at story level.

        Analyzes:
        1. Pairwise Spearman correlations: all stories, all artificial, all real, per model
        2. FP>1 group comparison (Mann-Whitney U for surprise and coherence)
        3. Mixed effects models with model as random effect (artificial stories only)

        Args:
            skip_non_converged: If True, skip models that did not converge in mixed effects.
            min_nsamp: Minimum number of samples required to include a story.
                       Stories with nsamp < min_nsamp are excluded (default 1 = include all).
                       Use 20 to exclude noisy nsamp=5 stories.
        """
        print("\n" + "=" * 80)
        print("METRIC CORRELATION ANALYSIS (Story-Level)")
        print("=" * 80 + "\n")
        print("Metrics (all × 25):")
        print("  Surprise    = (1 - mean(naive_true))")
        print("  Coherence   = mean(sampling_true)")
        print("  FP          = mean(sampling_true - naive_true)")
        print("  FP>1        = binary indicator (FP×25 > 1)")
        if min_nsamp > 1:
            print(f"  [Filter: nsamp >= {min_nsamp}]")
        print()

        # =====================================================================
        # STEP 1: Collect per-story data (deduplicate, prefer nsamp=20)
        # =====================================================================
        story_seen = {}  # story_key -> (llm_key, group, is_real)
        for key, value in self.llm_data.items():
            model_name, story_type, story_id = key
            nsamp = int(value['params'].get('nsamp', '5'))
            if nsamp < min_nsamp:
                continue
            is_real = story_type in ['poirot', 'sherlock']
            group = story_type if is_real else model_name
            story_key = f"{group}_{story_id}"
            if story_key not in story_seen or nsamp == 20:
                story_seen[story_key] = (key, group, is_real)

        records = []
        for story_key, (llm_key, group, is_real) in story_seen.items():
            model_name, story_type, story_id = llm_key

            sampling_dist, naive_dist, true_idx, _, _ = self.get_llm_distributions(
                model_name, story_type, story_id
            )
            if sampling_dist is None or naive_dist is None or true_idx is None:
                continue
            if len(sampling_dist) != len(naive_dist):
                continue

            naive_true = naive_dist[:, true_idx]
            sampling_true = sampling_dist[:, true_idx]
            surprise = (1 - np.mean(naive_true)) * 25
            coherence_llm = np.mean(sampling_true) * 25
            fp_llm = np.mean(sampling_true - naive_true) * 25
            fp_gt1 = int(fp_llm > 1)

            record = {
                'model': group,
                'story_id': story_id,
                'is_artificial': not is_real,
                'surprise': surprise,
                'coherence_llm': coherence_llm,
                'fp_llm': fp_llm,
                'fp_gt1': fp_gt1,
                'coherence_human': np.nan,
                'fp_human': np.nan,
                'fp_human_gt1': np.nan,
                'coherence_exp_0': np.nan,
                'coherence_exp_5': np.nan,
            }

            # Look up human data; gemini-2.5-flash tb-variants share human data
            human_group = group if 'gemini-2.5-flash' not in group else 'gemini-2.5-flash'
            human_story_name = f"{human_group} {story_id}"
            human_dist, _, _ = self.get_human_distribution(human_story_name)
            if human_dist is not None and len(human_dist) == len(naive_dist):
                human_true = human_dist[:, true_idx]
                record['coherence_human'] = np.mean(human_true) * 25
                record['fp_human'] = np.mean(human_true - naive_true) * 25
                record['fp_human_gt1'] = int(record['fp_human'] > 1)

            # Experienced reader coherence: 0-prev (setting 0) and 5-prev (setting 5)
            # Resolve true culprit by NAME to guard against suspect ordering differences
            # between the LLM data file and the experienced reader data file.
            llm_suspects = self.llm_data.get(llm_key, {}).get('data', {}).get('suspects', [])
            true_culprit_name = llm_suspects[true_idx] if true_idx < len(llm_suspects) else None

            for exp_setting, col in [(0, 'coherence_exp_0'), (5, 'coherence_exp_5')]:
                exp_dist, _ = self.get_experienced_reader_distributions(
                    model_name, story_type, story_id, setting=exp_setting
                )
                if exp_dist is None or len(exp_dist) == 0:
                    continue
                # Look up the true culprit's index in the experienced reader's suspect list
                exp_store = (self.experienced_reader_data if exp_setting <= 2
                             else self.experienced_reader_data_es3)
                exp_suspects = (exp_store.get(llm_key, {})
                                        .get('data', {})
                                        .get('suspects', []))
                if true_culprit_name and exp_suspects:
                    if true_culprit_name not in exp_suspects:
                        continue  # suspect name mismatch — skip rather than use wrong index
                    exp_true_idx = exp_suspects.index(true_culprit_name)
                else:
                    exp_true_idx = true_idx  # fallback: assume same ordering
                record[col] = np.mean(exp_dist[:, exp_true_idx]) * 25

            records.append(record)

        df = pd.DataFrame(records)
        n_art = int(df['is_artificial'].sum())
        n_real = int((~df['is_artificial']).sum())
        print(f"Total stories: {len(df)} ({n_art} artificial, {n_real} real)")
        print(f"Models: {sorted(df['model'].unique())}\n")

        results = {}

        # =====================================================================
        # PART 0: DIAGNOSTIC — experienced reader vs. know-it-all (upper bound)
        # =====================================================================
        print("=" * 80)
        print("PART 0: DIAGNOSTIC — Experienced Reader vs. Know-It-All Coherence")
        print("=" * 80)
        print("Theoretically, know-it-all (sampling) coherence is the upper bound.")
        print("Any story where exp_coherence > llm_coherence would violate this.\n")

        for exp_col, label in [('coherence_exp_0', '0-prev'), ('coherence_exp_5', '5-prev')]:
            sub = df[['model', 'coherence_llm', exp_col]].dropna()
            if len(sub) == 0:
                print(f"{label}: no data\n")
                continue
            n_exceeds = int((sub[exp_col] > sub['coherence_llm']).sum())
            n_total = len(sub)
            mean_diff = float((sub[exp_col] - sub['coherence_llm']).mean())
            median_diff = float((sub[exp_col] - sub['coherence_llm']).median())
            print(f"Exp ({label}) vs. Know-It-All  [n={n_total}]")
            print(f"  exp_coherence > llm_coherence: {n_exceeds}/{n_total} ({100*n_exceeds/n_total:.1f}%)")
            print(f"  Mean (exp − llm):   {mean_diff:+.3f}")
            print(f"  Median (exp − llm): {median_diff:+.3f}")
            # Per-model breakdown
            print(f"  Per-model mean (exp − llm) and mean llm_coherence:")
            for mdl, grp in sub.groupby('model'):
                diff = (grp[exp_col] - grp['coherence_llm']).mean()
                mean_llm = grp['coherence_llm'].mean()
                mean_exp = grp[exp_col].mean()
                print(f"    {mdl:<35s} diff={diff:+.3f}  llm={mean_llm:.3f}  exp={mean_exp:.3f}  n={len(grp)}")
            # Show top 5 worst violations
            if n_exceeds > 0:
                top_violations = (sub[exp_col] - sub['coherence_llm']).nlargest(5)
                print(f"  Top violations (exp − llm):")
                for idx, val in top_violations.items():
                    print(f"    story {idx} [{sub.loc[idx,'model']}]: {val:+.3f}  (llm={sub.loc[idx,'coherence_llm']:.3f}, exp={sub.loc[idx, exp_col]:.3f})")
            print()

        # =====================================================================
        # PART 1: PAIRWISE SPEARMAN CORRELATIONS
        # =====================================================================
        print("=" * 80)
        print("PART 1: PAIRWISE SPEARMAN CORRELATIONS")
        print("=" * 80)

        metric_pairs = [
            ('surprise',         'coherence_llm',    'Surprise vs. Coherence (LLM)'),
            ('surprise',         'coherence_human',  'Surprise vs. Coherence (Human)'),
            ('surprise',         'coherence_exp_0',  'Surprise vs. Coherence (Exp 0-prev)'),
            ('surprise',         'coherence_exp_5',  'Surprise vs. Coherence (Exp 5-prev)'),
            ('surprise',         'fp_llm',           'Surprise vs. FP (LLM)'),
            ('surprise',         'fp_human',         'Surprise vs. FP (Human)'),
            ('fp_llm',           'fp_human',         'FP (LLM) vs. FP (Human)'),
            ('coherence_llm',    'fp_llm',           'Coherence (LLM) vs. FP (LLM)'),
            ('coherence_llm',    'fp_human',         'Coherence (LLM) vs. FP (Human)'),
            ('coherence_human',  'fp_human',         'Coherence (Human) vs. FP (Human)'),
        ]

        def _corr_block(df_sub, n_min=5):
            sub_results = {}
            for v1, v2, label in metric_pairs:
                valid = df_sub[[v1, v2]].dropna()
                n = len(valid)
                if n >= n_min:
                    rho, p_rho = stats.spearmanr(valid[v1], valid[v2])
                    r, p_r = stats.pearsonr(valid[v1], valid[v2])
                    sig_rho = '***' if p_rho < 0.001 else '**' if p_rho < 0.01 else '*' if p_rho < 0.05 else 'ns'
                    sig_r   = '***' if p_r   < 0.001 else '**' if p_r   < 0.01 else '*' if p_r   < 0.05 else 'ns'
                    print(f"  {label}: Spearman r={rho:.3f} {sig_rho}, Pearson r={r:.3f} {sig_r} (n={n})")
                    sub_results[label] = {'spearman_r': rho, 'spearman_p': p_rho,
                                          'pearson_r': r,   'pearson_p': p_r, 'n': n}
                elif n > 0:
                    print(f"  {label}: n={n} (too few)")
            return sub_results

        results['correlations'] = {}

        for label, mask in [
            ('All stories',        pd.Series([True] * len(df), index=df.index)),
            ('Artificial stories', df['is_artificial']),
            ('Real stories',      ~df['is_artificial']),
        ]:
            print(f"\n{label} (n={int(mask.sum())}):")
            print("-" * 60)
            results['correlations'][label] = _corr_block(df[mask])

        print("\nPer model:")
        results['correlations']['per_model'] = {}
        for model in sorted(df['model'].unique()):
            df_m = df[df['model'] == model]
            print(f"\n  {model} (n={len(df_m)}):")
            results['correlations']['per_model'][model] = _corr_block(df_m, n_min=4)

        # =====================================================================
        # PART 2: FP>1 GROUP COMPARISON (Mann-Whitney U)
        # =====================================================================
        print("\n" + "=" * 80)
        print("PART 2: FP>1 GROUP COMPARISON (Mann-Whitney U)")
        print("  Do FP>1 stories differ in surprise/coherence from FP<=1 stories?")
        print("=" * 80)

        results['fp_gt1_comparison'] = {}

        for label, df_sub in [
            ('All stories',        df),
            ('Artificial stories', df[df['is_artificial']]),
            ('Real stories',       df[~df['is_artificial']]),
        ]:
            print(f"\n{label}:")
            fp1 = df_sub[df_sub['fp_gt1'] == 1]
            fp0 = df_sub[df_sub['fp_gt1'] == 0]
            print(f"  FP>1: n={len(fp1)}, FP<=1: n={len(fp0)}")
            sub = {}
            for metric, metric_label in [
                ('surprise',      'Surprise'),
                ('coherence_llm', 'Coherence (LLM)'),
            ]:
                g1 = fp1[metric].dropna().values
                g0 = fp0[metric].dropna().values
                if len(g1) >= 3 and len(g0) >= 3:
                    u_stat, p = stats.mannwhitneyu(g1, g0, alternative='two-sided')
                    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
                    print(f"  {metric_label}: FP>1 mean={np.mean(g1):.2f}, FP<=1 mean={np.mean(g0):.2f}"
                          f", U={u_stat:.0f}, p={p:.4f} {sig}")
                    sub[metric_label] = {'U': float(u_stat), 'p': p,
                                         'mean_fp1': float(np.mean(g1)),
                                         'mean_fp0': float(np.mean(g0))}
                else:
                    print(f"  {metric_label}: insufficient data")
            results['fp_gt1_comparison'][label] = sub

        # Per-model FP>1 ratio vs. mean surprise and coherence
        print("\nPer-model aggregates (artificial only):")
        df_art_agg = df[df['is_artificial']].groupby('model').agg(
            fp_gt1_ratio=('fp_gt1', 'mean'),
            mean_surprise=('surprise', 'mean'),
            mean_coherence=('coherence_llm', 'mean'),
            n=('surprise', 'count'),
        ).reset_index()
        print(df_art_agg.to_string(index=False))

        if len(df_art_agg) >= 5:
            r, p = stats.spearmanr(df_art_agg['mean_surprise'], df_art_agg['fp_gt1_ratio'])
            print(f"\nSpearman (model-level): mean_surprise vs. fp_gt1_ratio: r={r:.3f}, p={p:.4f}")
            r, p = stats.spearmanr(df_art_agg['mean_coherence'], df_art_agg['fp_gt1_ratio'])
            print(f"Spearman (model-level): mean_coherence vs. fp_gt1_ratio: r={r:.3f}, p={p:.4f}")
            r, p = stats.spearmanr(df_art_agg['mean_surprise'], df_art_agg['mean_coherence'])
            print(f"Spearman (model-level): mean_surprise vs. mean_coherence: r={r:.3f}, p={p:.4f}")
            results['fp_gt1_comparison']['model_level'] = df_art_agg.to_dict('records')

        # =====================================================================
        # PART 3: MIXED EFFECTS MODELS (model as random effect)
        # =====================================================================
        if run_mixed_effects:
            print("\n" + "=" * 80)
            print("PART 3: MIXED EFFECTS MODELS (Model as Random Effect)")
            print("  Controls for between-model variability; artificial stories only")
            print("=" * 80 + "\n")

            df_art_me = df[df['is_artificial']].copy()
            n_models_me = df_art_me['model'].nunique()
            print(f"Data: {len(df_art_me)} artificial stories across {n_models_me} models\n")

            results['mixed_models'] = {}

            print("Note: FP (LLM) = Coherence (LLM) + Surprise - 25 exactly, so LLM FP models")
            print("      are excluded as redundant. Core test: Coherence ~ Surprise (LLM & Human).\n")

            if len(df_art_me) < 20 or n_models_me < 3:
                print("Insufficient data for mixed effects models.\n")
            else:
                me_specs = [
                    ("coherence_llm ~ surprise",
                     "Coherence (LLM) ~ Surprise",
                     ["coherence_llm", "surprise"],
                     "surprise"),
                ]

                for col, lbl in [
                    ('coherence_human', 'Coherence (Human) ~ Surprise'),
                    ('coherence_exp_0', 'Coherence (Exp 0-prev) ~ Surprise'),
                    ('coherence_exp_5', 'Coherence (Exp 5-prev) ~ Surprise'),
                ]:
                    df_sub = df_art_me.dropna(subset=[col])
                    if len(df_sub) >= 20 and df_sub['model'].nunique() >= 3:
                        me_specs += [(f"{col} ~ surprise", lbl, [col, "surprise"], "surprise")]

                def _fit_mixed(formula, slope_var, fit_df):
                    try:
                        lm = smf.mixedlm(formula=formula, data=fit_df,
                                         groups=fit_df['model'],
                                         re_formula=f"~{slope_var}")
                        res = lm.fit(method='lbfgs')
                        if not np.isinf(res.llf) and res.bse.get('Intercept', 0) <= 1000:
                            return res, 'random_intercept_slope'
                    except Exception:
                        pass
                    try:
                        lm = smf.mixedlm(formula=formula, data=fit_df, groups=fit_df['model'])
                        res = lm.fit(method='lbfgs')
                        if not np.isinf(res.llf) and res.bse.get('Intercept', 0) <= 1000:
                            return res, 'random_intercept'
                    except Exception:
                        pass
                    outcome = formula.split('~')[0].strip()
                    predictors = formula.split('~')[1].strip()
                    res = smf.ols(f"{outcome} ~ {predictors} + C(model)", data=fit_df).fit()
                    return res, 'OLS_fixed_effects'

                method_labels = {
                    'random_intercept_slope': 'random intercept + slope',
                    'random_intercept':       'random intercept only',
                    'OLS_fixed_effects':      'OLS with model fixed effects',
                }

                for formula, lbl, cols, slope_var in me_specs:
                    print(f"Model: {lbl}")
                    print("-" * 60)
                    src = df_art_hum if 'fp_human' in cols else df_art_me
                    fit_df = src[['model'] + cols].dropna()
                    if len(fit_df) < 20 or fit_df['model'].nunique() < 3:
                        print("  Insufficient data\n")
                        continue
                    try:
                        res, method = _fit_mixed(formula, slope_var, fit_df)
                    except Exception as e:
                        print(f"  All fitting attempts failed: {e}\n")
                        continue
                    print(f"  [Method: {method_labels[method]}]")
                    print(res.summary())
                    entry = {
                        'n_obs': len(fit_df),
                        'n_models': fit_df['model'].nunique(),
                        'method': method,
                        'log_likelihood': getattr(res, 'llf', np.nan),
                        'params':  {k: v for k, v in res.params.items()
                                    if not k.startswith('C(model)')},
                        'pvalues': {k: v for k, v in res.pvalues.items()
                                    if not k.startswith('C(model)')},
                        'bse':     {k: v for k, v in res.bse.items()
                                    if not k.startswith('C(model)')},
                    }
                    results['mixed_models'][lbl] = entry
        else:
            print("[PART 3: Mixed effects skipped — run_mixed_effects=False]\n")

        # =====================================================================
        # PART 4: QUALITY HYPOTHESIS TEST
        #   H1: model quality increases both surprise and coherence, implying
        #       a positive between-model correlation of mean_S and mean_C.
        # =====================================================================
        from scipy.stats import norm as _norm

        print("\n" + "=" * 80)
        print("PART 4: QUALITY HYPOTHESIS TEST")
        print("  H1: better models write more surprising AND more solvable stories")
        print("  Tests: (a) between-model Spearman on per-model means,")
        print("         (b) story-level 95% CI + TOST (equivalence to zero)")
        print("=" * 80 + "\n")

        df_art = df[df['is_artificial']].copy()
        results['quality_hypothesis'] = {}

        # --- 4a. Between-model correlation of means ---
        model_agg = df_art.groupby('model').agg(
            mean_surprise=('surprise', 'mean'),
            mean_coherence_llm=('coherence_llm', 'mean'),
            mean_fp_llm=('fp_llm', 'mean'),
            n=('surprise', 'count'),
        ).reset_index()

        print("Between-model means (artificial stories only):")
        print(f"  {'Model':<35s} {'mean_S':>8} {'mean_C':>8} {'mean_FP':>8} {'n':>4}")
        for _, row in model_agg.iterrows():
            print(f"  {row['model']:<35s} {row['mean_surprise']:>8.2f} "
                  f"{row['mean_coherence_llm']:>8.2f} {row['mean_fp_llm']:>8.2f} {int(row['n']):>4}")

        print()
        bm_results = {}
        for x_col, y_col, lbl in [
            ('mean_surprise', 'mean_coherence_llm', 'mean_S vs. mean_C (LLM)'),
            ('mean_surprise', 'mean_fp_llm',        'mean_S vs. mean_FP (LLM)'),
        ]:
            valid = model_agg[[x_col, y_col]].dropna()
            n_m = len(valid)
            rho, p_two = stats.spearmanr(valid[x_col], valid[y_col])
            p_one = p_two / 2 if rho > 0 else 1.0 - p_two / 2
            sig = '***' if p_one < 0.001 else '**' if p_one < 0.01 else '*' if p_one < 0.05 else 'ns'
            print(f"Between-model Spearman ({lbl}):")
            print(f"  rho={rho:.3f}, p_two={p_two:.4f}, p_one_sided={p_one:.4f} {sig}  (n={n_m} models)")
            bm_results[lbl] = {'rho': rho, 'p_two': p_two, 'p_one': p_one, 'n': n_m}
        results['quality_hypothesis']['between_model'] = bm_results

        # --- 4b. Story-level TOST + 95% CI ---
        print()
        tost_results = {}
        for col, lbl in [
            ('coherence_llm',   'C_LLM'),
            ('coherence_human', 'C_Human'),
        ]:
            valid = df_art[['surprise', col]].dropna()
            n_sc = len(valid)
            if n_sc < 5:
                continue
            rho_sc, _ = stats.spearmanr(valid['surprise'], valid[col])

            # Fisher z CI
            z_obs = np.arctanh(np.clip(rho_sc, -0.9999, 0.9999))
            se_z = 1.0 / np.sqrt(n_sc - 3)
            ci_lo = np.tanh(z_obs - 1.96 * se_z)
            ci_hi = np.tanh(z_obs + 1.96 * se_z)

            print(f"Story-level Spearman rho(Surprise, {lbl}) = {rho_sc:.3f}  (n={n_sc})")
            print(f"  95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]")

            entry = {'rho': rho_sc, 'ci_lo': ci_lo, 'ci_hi': ci_hi, 'n': n_sc, 'tost': {}}
            for rho0 in [0.2, 0.3, 0.4]:
                # TOST: test H0: rho >= rho0  (one-sided upper)
                z0 = np.arctanh(rho0)
                t_stat = (z_obs - z0) / se_z
                p_tost = float(_norm.cdf(t_stat))
                sig_t = '***' if p_tost < 0.001 else '**' if p_tost < 0.01 else '*' if p_tost < 0.05 else 'ns'
                verdict = f'reject H0 (effect < {rho0})' if p_tost < 0.05 else f'fail to reject H0'
                print(f"  TOST rho0={rho0:.1f}: p={p_tost:.4f} {sig_t}  [{verdict}]")
                entry['tost'][rho0] = {'p': p_tost, 'sig': sig_t}
            print()
            tost_results[lbl] = entry
        results['quality_hypothesis']['story_level'] = tost_results

        print("=" * 80)
        print("END OF METRIC CORRELATION ANALYSIS")
        print("=" * 80 + "\n")

        return results, df

    def analyze_learning_curve(self, reader_stats_df, run_mixed_effects=True, skip_non_converged=True):
        """Analyze if readers improve over time."""
        df = reader_stats_df.sort_values(['user_id', 'timestamp']).copy()

        df["model"] = [s.split()[0] for s in df["story"]]

        df['story_order_overall'] = df.groupby('user_id').cumcount() + 1
        df['story_order_by_type'] = df.groupby(['user_id', 'is_artificial']).cumcount() + 1
        df['story_order_by_model'] = df.groupby(['user_id', 'model']).cumcount() + 1

        results = {}

        # =====================================================================
        # PART 1: Simple Spearman Correlations
        # =====================================================================

        print("\n" + "=" * 80)
        print("LEARNING CURVE ANALYSIS - PART 1: SPEARMAN CORRELATIONS")
        print("=" * 80 + "\n")

        if len(df) > 0:
            corr, p_value = stats.spearmanr(df['story_order_overall'],
                                            df['relative_performance'])
            order_stats = df.groupby('story_order_overall').agg({
                'relative_performance': ['mean', 'std', 'count']
            }).reset_index()

            print(f"Overall correlation: r = {corr:.3f}, p = {p_value:.4f}")

            results['overall'] = {
                'correlation': corr,
                'p_value': p_value,
                'order_stats': order_stats
            }

        for story_type in ['artificial', 'real']:
            df_type = df[df['is_artificial'] == (story_type == 'artificial')]

            if len(df_type) > 0:
                corr_overall, p_overall = stats.spearmanr(df_type['story_order_overall'],
                                                          df_type['relative_performance'])

                corr_type, p_type = stats.spearmanr(df_type['story_order_by_type'],
                                                    df_type['relative_performance'])

                order_stats_overall = df_type.groupby('story_order_overall').agg({
                    'relative_performance': ['mean', 'std', 'count']
                }).reset_index()

                order_stats_type = df_type.groupby('story_order_by_type').agg({
                    'relative_performance': ['mean', 'std', 'count']
                }).reset_index()

                print(f"\n{story_type.capitalize()} stories:")
                print(f"  Overall order: r = {corr_overall:.3f}, p = {p_overall:.4f}")
                print(f"  Within-type order: r = {corr_type:.3f}, p = {p_type:.4f}")

                results[story_type] = {
                    'correlation_overall': corr_overall,
                    'p_value_overall': p_overall,
                    'correlation_by_type': corr_type,
                    'p_value_by_type': p_type,
                    'order_stats_overall': order_stats_overall,
                    'order_stats_by_type': order_stats_type
                }

        model_results = {}
        for model in df['model'].dropna().unique():
            df_model = df[df['model'] == model]

            if len(df_model) > 5:
                corr_overall, p_overall = stats.spearmanr(df_model['story_order_overall'],
                                                          df_model['relative_performance'])

                corr_model, p_model = stats.spearmanr(df_model['story_order_by_model'],
                                                      df_model['relative_performance'])

                order_stats_model = df_model.groupby('story_order_by_model').agg({
                    'relative_performance': ['mean', 'std', 'count']
                }).reset_index()

                model_results[model] = {
                    'correlation_overall': corr_overall,
                    'p_value_overall': p_overall,
                    'correlation_by_model': corr_model,
                    'p_value_by_model': p_model,
                    'order_stats_by_model': order_stats_model
                }

        results['by_model'] = model_results

        # =====================================================================
        # PART 2: Mixed Effects Models
        # =====================================================================

        if not run_mixed_effects:
            print("\nSkipping Part 2 (mixed effects). Set run_mixed_effects=True to enable.")
            results['mixed_models'] = {}
            return results, df

        print("\n" + "=" * 80)
        print("LEARNING CURVE ANALYSIS - PART 2: MIXED EFFECTS MODELS")
        print("=" * 80 + "\n")

        print("These models account for:")
        print("  - Participant-level differences (some people are better at detective stories)")
        print("  - Story-level difficulty (some stories are harder)")
        print()

        results['mixed_models'] = {}

        # Filter to participants with at least 3 observations
        participant_counts = df.groupby('user_id').size()
        valid_participants = participant_counts[participant_counts >= 3].index
        df_filtered = df[df['user_id'].isin(valid_participants)]

        print(f"FILTERING: Keeping participants with ≥3 observations")
        print(f"  Original: {df['user_id'].nunique()} participants, {len(df)} observations")
        print(f"  Filtered: {df_filtered['user_id'].nunique()} participants, {len(df_filtered)} observations")
        print(f"  Removed: {df['user_id'].nunique() - df_filtered['user_id'].nunique()} participants\n")

        if len(df_filtered) < 30 or df_filtered['user_id'].nunique() < 5:
            print("  Insufficient data after filtering. Skipping mixed models.\n")
        else:
            # Model 1: Overall learning effect (all stories combined)
            print("-" * 80)
            print("Model 1: Overall Learning Effect (All Stories)")
            print("-" * 80)
            print("Formula: relative_performance ~ story_order_overall")
            print("Random effects: participant + story\n")

            model_data = df_filtered[['user_id', 'story', 'relative_performance', 'story_order_overall']].dropna()

            if len(model_data) >= 30 and model_data['user_id'].nunique() >= 5:
                try:
                    # Mixed model with crossed random effects (participant and story)
                    model = smf.mixedlm(
                        formula="relative_performance ~ story_order_overall",
                        data=model_data,
                        groups=model_data["user_id"],
                        re_formula="1",  # Random intercept for user
                        vc_formula={"story": "0 + C(story)"}  # Random intercept for story
                    )
                    result = model.fit(method='lbfgs')

                    log_likelihood = result.llf
                    intercept_se = result.bse['Intercept']
                    not_converged = np.isinf(log_likelihood) or intercept_se > 1000

                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model shows signs of convergence failure!")
                            print(f"   Log-likelihood: {log_likelihood}")
                            print(f"   Intercept SE: {intercept_se}")
                            print("   Results may be unreliable.\n")

                    results['mixed_models']['overall'] = {
                        'beta': result.params['story_order_overall'],
                        'se': result.bse['story_order_overall'],
                        'p_value': result.pvalues['story_order_overall'],
                        'n_obs': len(model_data),
                        'n_participants': model_data['user_id'].nunique(),
                        'n_stories': model_data['story'].nunique(),
                        'converged': result.converged,
                        'log_likelihood': log_likelihood
                    }
                except Exception as e:
                    print(f"  Model failed: {e}\n")
            else:
                print(f"  Insufficient data\n")

            # Model 2: Learning effect for artificial stories
            print("-" * 80)
            print("Model 2: Learning Effect (Artificial Stories Only)")
            print("-" * 80)

            artificial_data = df_filtered[df_filtered['is_artificial'] == True]

            # ADDITIONAL FILTERING: Only participants with ≥3 artificial stories
            participant_counts_art = artificial_data.groupby('user_id').size()
            valid_participants_art = participant_counts_art[participant_counts_art >= 3].index
            artificial_data_filtered = artificial_data[artificial_data['user_id'].isin(valid_participants_art)]

            print(f"  Additional filtering for artificial stories:")
            print(
                f"    Before: {artificial_data['user_id'].nunique()} participants, {len(artificial_data)} observations")
            print(
                f"    After:  {len(valid_participants_art)} participants, {len(artificial_data_filtered)} observations")
            print(f"    Removed: {artificial_data['user_id'].nunique() - len(valid_participants_art)} participants\n")

            model_data = artificial_data_filtered[
                ['user_id', 'story', 'relative_performance', 'story_order_by_type']].dropna()

            if len(model_data) >= 30 and model_data['user_id'].nunique() >= 5:
                from statsmodels.regression.linear_model import OLS

                # Approach 1: Story fixed effects only (assume no participant differences)
                print("  Approach 1: Story Fixed Effects Only")
                print("  " + "-" * 76)
                print("  Assumes: Stories differ, participants are homogeneous\n")
                try:
                    model = OLS.from_formula(
                        "relative_performance ~ story_order_by_type + C(story)",
                        data=model_data
                    )
                    result = model.fit()

                    print(result.summary())
                    print("\n✓ Model completed.\n")

                    results['mixed_models']['artificial_story_only'] = {
                        'approach': 'Story fixed effects only',
                        'beta': result.params['story_order_by_type'],
                        'se': result.bse['story_order_by_type'],
                        'p_value': result.pvalues['story_order_by_type'],
                        'n_obs': len(model_data),
                        'n_participants': model_data['user_id'].nunique(),
                        'n_stories': model_data['story'].nunique()
                    }
                except Exception as e:
                    print(f"  Model failed: {e}\n")

                # Approach 2: Participant random effects only (assume no story differences)
                print("  Approach 2: Participant Random Effects Only")
                print("  " + "-" * 76)
                print("  Assumes: Participants differ, stories are homogeneous\n")
                try:
                    model = smf.mixedlm(
                        formula="relative_performance ~ story_order_by_type",
                        data=model_data,
                        groups=model_data["user_id"]
                    )
                    result = model.fit(method='lbfgs')

                    not_converged = not result.converged or np.isinf(result.llf)
                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model did not converge properly!\n")
                        else:
                            print("\n✓ Model converged successfully.\n")

                    results['mixed_models']['artificial_participant_only'] = {
                        'approach': 'Participant random effects only',
                        'beta': result.params['story_order_by_type'],
                        'se': result.bse['story_order_by_type'],
                        'p_value': result.pvalues['story_order_by_type'],
                        'n_obs': len(model_data),
                        'n_participants': model_data['user_id'].nunique(),
                        'n_stories': model_data['story'].nunique(),
                        'converged': result.converged
                    }
                except Exception as e:
                    print(f"  Model failed: {e}\n")

                # Approach 3: Use relative_performance (already story-adjusted) with participant random effects
                print("  Approach 3: Relative Performance (story-adjusted) ~ Order")
                print("  " + "-" * 76)
                print("  Note: relative_performance is ALREADY adjusted for story difficulty")
                print("  Just test if it changes with story order, accounting for participant differences\n")
                try:
                    model = smf.mixedlm(
                        formula="relative_performance ~ story_order_by_type",
                        data=model_data,
                        groups=model_data["user_id"]
                    )
                    result = model.fit(method='lbfgs')

                    not_converged = not result.converged or np.isinf(result.llf)
                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model did not converge properly!\n")
                        else:
                            print("\n✓ Model converged successfully.\n")

                    results['mixed_models']['artificial_relative_perf'] = {
                        'approach': 'Relative performance with participant RE',
                        'beta': result.params['story_order_by_type'],
                        'se': result.bse['story_order_by_type'],
                        'p_value': result.pvalues['story_order_by_type'],
                        'n_obs': len(model_data),
                        'n_participants': model_data['user_id'].nunique(),
                        'n_stories': model_data['story'].nunique(),
                        'converged': result.converged
                    }
                except Exception as e:
                    print(f"  Model failed: {e}\n")

            else:
                print(
                    f"  Insufficient data after filtering (n={len(model_data)}, groups={model_data['user_id'].nunique()})\n")

            # Model 3: Learning effect for real stories
            print("-" * 80)
            print("Model 3: Learning Effect (Real Stories Only)")
            print("-" * 80)

            real_data = df_filtered[df_filtered['is_artificial'] == False]

            # ADDITIONAL FILTERING: Only participants with ≥3 real stories
            participant_counts_real = real_data.groupby('user_id').size()
            valid_participants_real = participant_counts_real[participant_counts_real >= 3].index
            real_data_filtered = real_data[real_data['user_id'].isin(valid_participants_real)]

            print(f"  Additional filtering for real stories:")
            print(f"    Before: {real_data['user_id'].nunique()} participants, {len(real_data)} observations")
            print(f"    After:  {len(valid_participants_real)} participants, {len(real_data_filtered)} observations")
            print(f"    Removed: {real_data['user_id'].nunique() - len(valid_participants_real)} participants\n")

            model_data = real_data_filtered[
                ['user_id', 'story', 'relative_performance', 'story_order_by_type']].dropna()

            if len(model_data) >= 30 and model_data['user_id'].nunique() >= 5:
                from statsmodels.regression.linear_model import OLS

                # Approach 1: Story fixed effects only (assume no participant differences)
                print("  Approach 1: Story Fixed Effects Only")
                print("  " + "-" * 76)
                print("  Assumes: Stories differ, participants are homogeneous\n")
                try:
                    model = OLS.from_formula(
                        "relative_performance ~ story_order_by_type + C(story)",
                        data=model_data
                    )
                    result = model.fit()

                    print(result.summary())
                    print("\n✓ Model completed.\n")

                    results['mixed_models']['real_story_only'] = {
                        'approach': 'Story fixed effects only',
                        'beta': result.params['story_order_by_type'],
                        'se': result.bse['story_order_by_type'],
                        'p_value': result.pvalues['story_order_by_type'],
                        'n_obs': len(model_data),
                        'n_participants': model_data['user_id'].nunique(),
                        'n_stories': model_data['story'].nunique()
                    }
                except Exception as e:
                    print(f"  Model failed: {e}\n")

                # Approach 2: Participant random effects only (assume no story differences)
                print("  Approach 2: Participant Random Effects Only")
                print("  " + "-" * 76)
                print("  Assumes: Participants differ, stories are homogeneous\n")
                try:
                    model = smf.mixedlm(
                        formula="relative_performance ~ story_order_by_type",
                        data=model_data,
                        groups=model_data["user_id"]
                    )
                    result = model.fit(method='lbfgs')

                    not_converged = not result.converged or np.isinf(result.llf)
                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model did not converge properly!\n")
                        else:
                            print("\n✓ Model converged successfully.\n")

                    results['mixed_models']['real_participant_only'] = {
                        'approach': 'Participant random effects only',
                        'beta': result.params['story_order_by_type'],
                        'se': result.bse['story_order_by_type'],
                        'p_value': result.pvalues['story_order_by_type'],
                        'n_obs': len(model_data),
                        'n_participants': model_data['user_id'].nunique(),
                        'n_stories': model_data['story'].nunique(),
                        'converged': result.converged
                    }
                except Exception as e:
                    print(f"  Model failed: {e}\n")

                # Approach 3: Use relative_performance (already story-adjusted) with participant random effects
                print("  Approach 3: Relative Performance (story-adjusted) ~ Order")
                print("  " + "-" * 76)
                print("  Note: relative_performance is ALREADY adjusted for story difficulty")
                print("  Just test if it changes with story order, accounting for participant differences\n")
                try:
                    model = smf.mixedlm(
                        formula="relative_performance ~ story_order_by_type",
                        data=model_data,
                        groups=model_data["user_id"]
                    )
                    result = model.fit(method='lbfgs')

                    not_converged = not result.converged or np.isinf(result.llf)
                    if skip_non_converged and not_converged:
                        print("  [SKIPPED: model did not converge]\n")
                    else:
                        print(result.summary())
                        if not_converged:
                            print("\n⚠️  WARNING: Model did not converge properly!\n")
                        else:
                            print("\n✓ Model converged successfully.\n")

                    results['mixed_models']['real_relative_perf'] = {
                        'approach': 'Relative performance with participant RE',
                        'beta': result.params['story_order_by_type'],
                        'se': result.bse['story_order_by_type'],
                        'p_value': result.pvalues['story_order_by_type'],
                        'n_obs': len(model_data),
                        'n_participants': model_data['user_id'].nunique(),
                        'n_stories': model_data['story'].nunique(),
                        'converged': result.converged
                    }
                except Exception as e:
                    print(f"  Model failed: {e}\n")

            else:
                print(
                    f"  Insufficient data after filtering (n={len(model_data)}, groups={model_data['user_id'].nunique()})\n")

        print("=" * 80)
        print("END OF LEARNING CURVE ANALYSIS")
        print("=" * 80 + "\n")

        return results, df


    def analyze_subjective_mixed_effects(self):
        """
        Analyze subjective ratings using mixed effects models to compare
        artificial vs real stories, accounting for annotator and story effects.

        For each rating dimension, fits:
            rating ~ is_artificial + (1|story) + (1|participant_id)

        Story is used as the primary grouping factor (each story has multiple
        raters), and participant is added as a variance component to account
        for annotator-level preferences. This structure matches the data:
        many stories with a handful of raters each, and raters who may
        evaluate multiple stories.

        Returns:
        --------
        dict: Results for each rating dimension containing model fits, coefficients, p-values, ICCs
        pd.DataFrame: The individual-level data used for modeling
        """
        print("\n" + "=" * 80)
        print("MIXED EFFECTS ANALYSIS: SUBJECTIVE RATINGS (ARTIFICIAL vs REAL)")
        print("Accounting for annotator and story random effects")
        print("=" * 80 + "\n")

        rating_cols_map = {
            'familiarity': 'familiarity', 'effort': 'effort', 'surprising': 'surprise',
            'fair': 'fairness', 'coherent': 'coherence', 'enjoyable': 'enjoyability'
        }
        rating_dims = ['familiarity', 'effort', 'surprise', 'fairness', 'coherence', 'enjoyability']

        # Collect individual-level data
        individual_data = []

        for story_name, data in self.human_data.items():
            df = data['df']
            if len(df) == 0:
                continue

            model_key, story_id = story_name.split()
            is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

            rating_col_names = df.columns[-6:]

            for idx, row in df.iterrows():
                participant_id = row[df.columns[1]]
                participant_ratings = {}

                for col in rating_col_names:
                    col_lower = col.lower()
                    for keyword, dim_name in rating_cols_map.items():
                        if keyword in col_lower:
                            participant_ratings[dim_name] = row[col]
                            break

                if not participant_ratings or all(pd.isna(v) for v in participant_ratings.values()):
                    continue

                individual_data.append({
                    'participant_id': str(participant_id),
                    'story': story_name,
                    'model': model_key,
                    'is_artificial': 1 if is_artificial else 0,
                    'is_artificial_label': 'Artificial' if is_artificial else 'Real',
                    **participant_ratings
                })

        df_all = pd.DataFrame(individual_data)

        if len(df_all) == 0:
            print("No rating data found!")
            return {}, pd.DataFrame()

        n_participants = df_all['participant_id'].nunique()
        n_stories = df_all['story'].nunique()
        obs_per_story = df_all.groupby('story').size()
        obs_per_participant = df_all.groupby('participant_id').size()

        print(f"Data: {len(df_all)} observations from {n_participants} participants "
              f"across {n_stories} stories")
        print(f"  Artificial stories: {df_all[df_all['is_artificial'] == 1]['story'].nunique()}")
        print(f"  Real stories: {df_all[df_all['is_artificial'] == 0]['story'].nunique()}")
        print(f"  Observations per story:       min={obs_per_story.min()}, "
              f"median={obs_per_story.median():.0f}, max={obs_per_story.max()}")
        print(f"  Observations per participant: min={obs_per_participant.min()}, "
              f"median={obs_per_participant.median():.0f}, max={obs_per_participant.max()}")
        print()

        results = {}

        for dim in rating_dims:
            if dim not in df_all.columns:
                continue

            dim_data = df_all[['participant_id', 'story', 'is_artificial', dim]].dropna()

            if len(dim_data) < 30:
                print(f"{dim.upper()}: Insufficient data (n={len(dim_data)})")
                results[dim] = {'converged': False, 'error': 'Insufficient data', 'n_obs': len(dim_data)}
                continue

            print(f"{'-' * 70}")
            print(f"{dim.upper()}")
            print(f"{'-' * 70}")

            # ---- Approach: story as primary group, participant as variance component ----
            # This is the natural choice: stories have multiple raters (good group sizes),
            # and participants who rate multiple stories contribute a crossed effect.
            try:
                vc = {'participant': '0 + C(participant_id)'}
                model = smf.mixedlm(
                    formula=f"{dim} ~ is_artificial",
                    data=dim_data,
                    groups=dim_data["story"],
                    vc_formula=vc
                )
                result = model.fit(method='lbfgs', reml=True)

                print(result.summary())

                beta = result.params['is_artificial']
                se = result.bse['is_artificial']
                p_value = result.pvalues['is_artificial']
                ci_lower, ci_upper = result.conf_int().loc['is_artificial']
                intercept = result.params['Intercept']

                # Extract variance components safely
                story_var = result.cov_re.iloc[0, 0] if result.cov_re.size > 0 else 0.0
                residual_var = result.scale

                participant_var = 0.0
                if hasattr(result, 'vcomp') and result.vcomp is not None:
                    for key in result.vcomp:
                        participant_var += result.vcomp[key]

                total_var = story_var + participant_var + residual_var
                icc_story = story_var / total_var if total_var > 0 else 0
                icc_participant = participant_var / total_var if total_var > 0 else 0

                # Means by group (estimated from model)
                real_mean = intercept
                artificial_mean = intercept + beta

                print(f"\n  KEY RESULTS (story RE + participant VC):")
                print(f"  Fixed effect of is_artificial (Artificial - Real):")
                print(f"    beta = {beta:.3f} (SE = {se:.3f})")
                print(f"    95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                print(f"    p-value: {p_value:.6f} "
                      f"{'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'}")
                print(f"  Estimated means: Real={real_mean:.3f}, Artificial={artificial_mean:.3f}")
                print(f"  Variance components:")
                print(f"    Story:       {story_var:.3f} (ICC={icc_story:.3f}, {icc_story * 100:.1f}%)")
                print(f"    Participant: {participant_var:.3f} (ICC={icc_participant:.3f}, {icc_participant * 100:.1f}%)")
                print(f"    Residual:    {residual_var:.3f} ({residual_var / total_var * 100:.1f}%)" if total_var > 0 else "")
                print(f"  Sample: {len(dim_data)} obs, {dim_data['participant_id'].nunique()} participants, "
                      f"{dim_data['story'].nunique()} stories")
                print()

                results[dim] = {
                    'converged': True,
                    'model_type': 'crossed',
                    'fitted_model': result,
                    'beta': beta,
                    'se': se,
                    'p_value': p_value,
                    'ci': (ci_lower, ci_upper),
                    'intercept': intercept,
                    'real_mean_est': real_mean,
                    'artificial_mean_est': artificial_mean,
                    'story_var': story_var,
                    'participant_var': participant_var,
                    'residual_var': residual_var,
                    'icc_story': icc_story,
                    'icc_participant': icc_participant,
                    'n_obs': len(dim_data),
                    'n_participants': dim_data['participant_id'].nunique(),
                    'n_stories': dim_data['story'].nunique()
                }

            except Exception as e:
                print(f"  Crossed random effects model failed: {e}")
                print(f"  Falling back to story-only random intercept model...")

                try:
                    model_fallback = smf.mixedlm(
                        formula=f"{dim} ~ is_artificial",
                        data=dim_data,
                        groups=dim_data["story"]
                    )
                    result_fb = model_fallback.fit(method='lbfgs')

                    print(result_fb.summary())

                    beta = result_fb.params['is_artificial']
                    se = result_fb.bse['is_artificial']
                    p_value = result_fb.pvalues['is_artificial']
                    ci_lower, ci_upper = result_fb.conf_int().loc['is_artificial']
                    intercept = result_fb.params['Intercept']

                    story_var = result_fb.cov_re.iloc[0, 0] if result_fb.cov_re.size > 0 else 0.0
                    residual_var = result_fb.scale
                    total_var = story_var + residual_var
                    icc_story = story_var / total_var if total_var > 0 else 0

                    print(f"\n  KEY RESULTS (story-only random intercept):")
                    print(f"  beta = {beta:.3f} (SE = {se:.3f}), p = {p_value:.6f}")
                    print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                    print(f"  Estimated means: Real={intercept:.3f}, Artificial={intercept + beta:.3f}")
                    print(f"  ICC (story): {icc_story:.3f}")
                    print()

                    results[dim] = {
                        'converged': True,
                        'model_type': 'story_only',
                        'fitted_model': result_fb,
                        'beta': beta,
                        'se': se,
                        'p_value': p_value,
                        'ci': (ci_lower, ci_upper),
                        'intercept': intercept,
                        'real_mean_est': intercept,
                        'artificial_mean_est': intercept + beta,
                        'story_var': story_var,
                        'residual_var': residual_var,
                        'icc_story': icc_story,
                        'n_obs': len(dim_data),
                        'n_participants': dim_data['participant_id'].nunique(),
                        'n_stories': dim_data['story'].nunique()
                    }

                except Exception as e2:
                    print(f"  Story-only fallback also failed: {e2}")
                    print(f"  Falling back to participant-only random intercept model...")

                    try:
                        model_participant = smf.mixedlm(
                            formula=f"{dim} ~ is_artificial",
                            data=dim_data,
                            groups=dim_data["participant_id"]
                        )
                        result_pt = model_participant.fit(method='lbfgs')

                        print(result_pt.summary())

                        beta = result_pt.params['is_artificial']
                        se = result_pt.bse['is_artificial']
                        p_value = result_pt.pvalues['is_artificial']
                        ci_lower, ci_upper = result_pt.conf_int().loc['is_artificial']
                        intercept = result_pt.params['Intercept']

                        participant_var = result_pt.cov_re.iloc[0, 0] if result_pt.cov_re.size > 0 else 0.0
                        residual_var = result_pt.scale
                        total_var = participant_var + residual_var
                        icc_participant = participant_var / total_var if total_var > 0 else 0

                        print(f"\n  KEY RESULTS (participant-only random intercept):")
                        print(f"  beta = {beta:.3f} (SE = {se:.3f}), p = {p_value:.6f}")
                        print(f"  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")
                        print(f"  Estimated means: Real={intercept:.3f}, Artificial={intercept + beta:.3f}")
                        print(f"  ICC (participant): {icc_participant:.3f}")
                        print()

                        results[dim] = {
                            'converged': True,
                            'model_type': 'participant_only',
                            'fitted_model': result_pt,
                            'beta': beta,
                            'se': se,
                            'p_value': p_value,
                            'ci': (ci_lower, ci_upper),
                            'intercept': intercept,
                            'real_mean_est': intercept,
                            'artificial_mean_est': intercept + beta,
                            'participant_var': participant_var,
                            'residual_var': residual_var,
                            'icc_participant': icc_participant,
                            'n_obs': len(dim_data),
                            'n_participants': dim_data['participant_id'].nunique(),
                            'n_stories': dim_data['story'].nunique()
                        }

                    except Exception as e3:
                        print(f"  Participant-only fallback also failed: {e3}")
                        results[dim] = {'converged': False, 'error': str(e3), 'n_obs': len(dim_data)}

        # =========================================================================
        # SUMMARY TABLE
        # =========================================================================
        print("\n" + "=" * 80)
        print("SUMMARY: Mixed Effects Models for Artificial vs Real")
        print("=" * 80 + "\n")

        summary_rows = []
        for dim in rating_dims:
            if dim in results and results[dim].get('converged'):
                r = results[dim]
                sig = '***' if r['p_value'] < 0.001 else '**' if r['p_value'] < 0.01 else '*' if r['p_value'] < 0.05 else 'ns'
                row = {
                    'Dimension': dim.capitalize(),
                    'Model': r.get('model_type', 'crossed'),
                    'beta (Art-Real)': f"{r['beta']:.3f}",
                    'SE': f"{r['se']:.3f}",
                    'p-value': f"{r['p_value']:.4f}",
                    'Sig': sig,
                    'Real (est)': f"{r['real_mean_est']:.3f}",
                    'Art (est)': f"{r['artificial_mean_est']:.3f}",
                    'ICC_story': f"{r['icc_story']:.3f}" if 'icc_story' in r else 'N/A',
                    'ICC_part': f"{r['icc_participant']:.3f}" if 'icc_participant' in r else 'N/A',
                    'n': r['n_obs']
                }
                summary_rows.append(row)

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            print(summary_df.to_string(index=False))
        else:
            print("No models converged.")

        print("\n" + "=" * 80)
        print("END OF MIXED EFFECTS SUBJECTIVE ANALYSIS")
        print("=" * 80 + "\n")

        return results, df_all


    def print_subjective_score_summary(self):
        """Print a clean summary of subjective scores that properly accounts for
        the two sources of variability: different annotators and different stories.

        Reports:
        1. Story-level means (average rating per story, then summarize across stories)
           - This treats each story as one observation, avoiding pseudo-replication
        2. Within-story annotator agreement (how much do annotators disagree on the same story)
        3. Between-story variability (how much do stories differ)

        Grouped by: all stories, artificial only, real only.
        """
        print("\n" + "=" * 80)
        print("SUBJECTIVE SCORE SUMMARY")
        print("(Properly accounting for annotator and story variability)")
        print("=" * 80)

        rating_cols_map = {
            'familiarity': 'familiarity', 'effort': 'effort', 'surprising': 'surprise',
            'fair': 'fairness', 'coherent': 'coherence', 'enjoyable': 'enjoyability'
        }
        rating_dims = ['familiarity', 'effort', 'surprise', 'fairness', 'coherence', 'enjoyability']

        # Collect individual-level data
        individual_data = []
        for story_name, data in self.human_data.items():
            df = data['df']
            if len(df) == 0:
                continue
            model_key, story_id = story_name.split()
            is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))
            rating_col_names = df.columns[-6:]

            for idx, row in df.iterrows():
                participant_id = row[df.columns[1]]
                participant_ratings = {}
                for col in rating_col_names:
                    col_lower = col.lower()
                    for keyword, dim_name in rating_cols_map.items():
                        if keyword in col_lower:
                            participant_ratings[dim_name] = row[col]
                            break
                if not participant_ratings or all(pd.isna(v) for v in participant_ratings.values()):
                    continue
                individual_data.append({
                    'participant_id': str(participant_id),
                    'story': story_name,
                    'model': model_key,
                    'is_artificial': is_artificial,
                    **participant_ratings
                })

        df_all = pd.DataFrame(individual_data)
        if len(df_all) == 0:
            print("No data!")
            return

        # Compute per-story statistics
        story_stats = df_all.groupby(['story', 'is_artificial']).agg(
            **{f'{dim}_mean': (dim, 'mean') for dim in rating_dims if dim in df_all.columns},
            **{f'{dim}_std': (dim, 'std') for dim in rating_dims if dim in df_all.columns},
            n_annotators=('participant_id', 'nunique')
        ).reset_index()

        for label, subset_filter in [("ALL STORIES", None),
                                      ("ARTIFICIAL STORIES", True),
                                      ("REAL STORIES", False)]:
            if subset_filter is None:
                subset = story_stats
            else:
                subset = story_stats[story_stats['is_artificial'] == subset_filter]

            if len(subset) == 0:
                continue

            print(f"\n{'-' * 70}")
            print(f"{label} (n = {len(subset)} stories, "
                  f"total observations = {df_all[df_all['is_artificial'] == subset_filter].shape[0] if subset_filter is not None else len(df_all)})")
            print(f"{'-' * 70}")
            print(f"  {'Dimension':<15} {'Story Mean':>12} {'Between-story SD':>18} {'Within-story SD':>18}")
            print(f"  {'':.<15} {'(avg of means)':>12} {'(SD of story means)':>18} {'(avg of story SDs)':>18}")

            for dim in rating_dims:
                mean_col = f'{dim}_mean'
                std_col = f'{dim}_std'
                if mean_col not in subset.columns:
                    continue

                story_means = subset[mean_col].dropna()
                story_sds = subset[std_col].dropna()

                if len(story_means) == 0:
                    continue

                # Grand mean = mean of story means (each story weighted equally)
                grand_mean = story_means.mean()
                # Between-story SD = how much stories differ from each other
                between_sd = story_means.std()
                # Within-story SD = average of per-story SDs (annotator disagreement)
                within_sd = story_sds.mean()

                print(f"  {dim.capitalize():<15} {grand_mean:>10.2f}/5 {between_sd:>16.2f} {within_sd:>16.2f}")

        print(f"\n  Notes:")
        print(f"  - 'Story Mean' averages each story's mean rating (stories weighted equally)")
        print(f"  - 'Between-story SD' = variability of story means (story quality differences)")
        print(f"  - 'Within-story SD' = average annotator disagreement within a story")

        print("\n" + "=" * 80)
        print("END OF SUBJECTIVE SCORE SUMMARY")
        print("=" * 80 + "\n")


    def report_subjective_ratings_by_model(self, use_centering=True, use_zscore=False):
        """
        Report average subjective ratings by model/story type with normalization.
        Normalization is done per-dimension: each participant's mean for a given
        rating dimension (e.g., fairness) is computed across all their stories
        (both artificial and real), then subtracted.

        Args:
            use_centering: If True, report within-subject centered ratings
            use_zscore: If True, report within-subject z-scored ratings

        Returns:
            DataFrame with ratings by model
        """
        rating_cols = ['familiarity', 'effort', 'surprise', 'fairness', 'coherence', 'enjoyability']

        # Collect individual-level data
        individual_data = []

        for story_name, data in self.human_data.items():
            df = data['df']
            ratings = data['ratings']

            if len(ratings) == 0:
                continue

            model_key, story_id = story_name.split()
            is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

            # Extract rating column names
            rating_col_names = df.columns[-6:]

            # Loop through each participant
            for idx, row in df.iterrows():
                participant_id = row[df.columns[1]]
                participant_ratings = {}

                # Extract this participant's ratings
                for col in rating_col_names:
                    col_lower = col.lower()
                    if 'familiarity' in col_lower:
                        participant_ratings['familiarity'] = row[col]
                    elif 'effort' in col_lower:
                        participant_ratings['effort'] = row[col]
                    elif 'surprising' in col_lower:
                        participant_ratings['surprise'] = row[col]
                    elif 'fair' in col_lower:
                        participant_ratings['fairness'] = row[col]
                    elif 'coherent' in col_lower:
                        participant_ratings['coherence'] = row[col]
                    elif 'enjoyable' in col_lower:
                        participant_ratings['enjoyability'] = row[col]

                # Skip if participant didn't provide ratings
                if not participant_ratings or all(pd.isna(v) for v in participant_ratings.values()):
                    continue

                # Add row for this participant
                individual_data.append({
                    'participant_id': participant_id,
                    'story': story_name,
                    'model': model_key,
                    'is_artificial': is_artificial,
                    **participant_ratings
                })

        df_ratings = pd.DataFrame(individual_data)

        if len(df_ratings) == 0:
            print("No rating data found!")
            return None

        # Calculate person-level statistics for normalization
        # Per-dimension: for each rating column, compute each participant's mean and std
        # across ALL their stories (no separation between artificial and real)
        for col in rating_cols:
            if col not in df_ratings.columns:
                continue

            # Compute per-participant mean and std for this dimension
            person_col_stats = {}
            for pid in df_ratings['participant_id'].unique():
                person_data = df_ratings[df_ratings['participant_id'] == pid]
                col_values = person_data[col].dropna().values
                if len(col_values) > 0:
                    person_col_stats[pid] = {
                        'mean': np.mean(col_values),
                        'std': np.std(col_values) if len(col_values) > 1 else 0
                    }

            if use_centering:
                centered_col = f'{col}_centered'
                df_ratings[centered_col] = df_ratings.apply(
                    lambda row: row[col] - person_col_stats.get(row['participant_id'], {}).get('mean', 0)
                    if pd.notna(row[col]) else np.nan,
                    axis=1
                )

            if use_zscore:
                zscore_col = f'{col}_zscore'
                df_ratings[zscore_col] = df_ratings.apply(
                    lambda row: (row[col] - person_col_stats.get(row['participant_id'], {}).get('mean', 0)) /
                                person_col_stats.get(row['participant_id'], {}).get('std', 1)
                    if pd.notna(row[col]) and person_col_stats.get(row['participant_id'], {}).get('std', 0) > 0
                    else np.nan,
                    axis=1
                )

        return df_ratings


    def print_ratings_by_model(self):
        """Print comprehensive subjective ratings report by model."""

        print("\n" + "=" * 80)
        print("SUBJECTIVE RATINGS BY MODEL")
        print("(Per-dimension normalization across all stories)")
        print("=" * 80)

        # Get ratings data
        df_ratings = self.report_subjective_ratings_by_model(use_centering=True, use_zscore=True)

        if df_ratings is None:
            return

        rating_cols = ['familiarity', 'effort', 'surprise', 'fairness', 'coherence', 'enjoyability']

        # Report by story type (artificial vs real)
        for story_type in ["ARTIFICIAL STORIES", "REAL STORIES"]:
            print(f"\n{'-' * 80}")
            print(f"{story_type}")
            print(f"{'-' * 80}")

            is_artificial = (story_type == "ARTIFICIAL STORIES")
            df_type = df_ratings[df_ratings['is_artificial'] == is_artificial]

            if len(df_type) == 0:
                print("  No data available")
                continue

            # Group by model
            for model in sorted(df_type['model'].unique()):
                df_model = df_type[df_type['model'] == model]

                print(f"\n{model}:")
                print(f"  Number of ratings: {len(df_model)}")

                for rating_col in rating_cols:
                    if rating_col not in df_model.columns:
                        continue

                    # Raw ratings
                    raw_data = df_model[rating_col].dropna()
                    if len(raw_data) == 0:
                        continue

                    raw_mean = raw_data.mean()
                    raw_std = raw_data.std()

                    # Centered ratings (no category suffix)
                    centered_col = f'{rating_col}_centered'
                    if centered_col in df_model.columns:
                        centered_data = df_model[centered_col].dropna()
                        centered_mean = centered_data.mean() if len(centered_data) > 0 else np.nan
                        centered_std = centered_data.std() if len(centered_data) > 0 else np.nan
                    else:
                        centered_mean = np.nan
                        centered_std = np.nan

                    # Z-scored ratings (no category suffix)
                    zscore_col = f'{rating_col}_zscore'
                    if zscore_col in df_model.columns:
                        zscore_data = df_model[zscore_col].dropna()
                        zscore_mean = zscore_data.mean() if len(zscore_data) > 0 else np.nan
                        zscore_std = zscore_data.std() if len(zscore_data) > 0 else np.nan
                    else:
                        zscore_mean = np.nan
                        zscore_std = np.nan

                    print(f"  {rating_col.capitalize()}:")
                    print(f"    Raw:      Mean={raw_mean:.3f}/5, Std={raw_std:.3f}, n={len(raw_data)}")
                    if not np.isnan(centered_mean):
                        print(f"    Centered: Mean={centered_mean:+.3f}, Std={centered_std:.3f}")
                    if not np.isnan(zscore_mean):
                        print(f"    Z-score:  Mean={zscore_mean:+.3f}, Std={zscore_std:.3f}")

        # =====================================================================
        # AGGREGATE SUBJECTIVE SCORES (Mean ± SD over individual stories)
        # =====================================================================
        print(f"\n{'-' * 80}")
        print("AGGREGATE SUBJECTIVE SCORES (Mean ± SD over individual stories)")
        print("  STD is computed over per-story mean ratings, NOT over individual responses")
        print(f"{'-' * 80}")

        # Compute per-story mean ratings
        story_means = df_ratings.groupby(['story', 'model', 'is_artificial'])[rating_cols].mean().reset_index()

        for label, subset_filter in [("ALL STORIES", None),
                                      ("ARTIFICIAL STORIES", True),
                                      ("REAL STORIES", False)]:
            if subset_filter is None:
                subset = story_means
            else:
                subset = story_means[story_means['is_artificial'] == subset_filter]

            if len(subset) == 0:
                continue

            print(f"\n  {label} (n={len(subset)} stories):")
            for rating_col in rating_cols:
                col_data = subset[rating_col].dropna()
                if len(col_data) > 0:
                    print(f"    {rating_col.capitalize():15s}: {col_data.mean():.3f} ± {col_data.std():.3f}  (n={len(col_data)})")

        print("\n" + "=" * 80)
        print("END OF SUBJECTIVE RATINGS REPORT")
        print("=" * 80 + "\n")

    # =========================================================================
    # Experienced reader analysis (NEW)
    # =========================================================================

    def print_experienced_reader_scores(self):
        """Print fairplay, surprise, and coherence scores for the experienced reader
        across all six settings (es2: 0-shot, 9-prev, 3-prev; es3: 0-shot-alt, 2-prev, 5-prev).

        Reports scores analogous to the naive/know-it-all reader scores.
        """
        if not self.experienced_reader_data and not self.experienced_reader_data_es3:
            print("No experienced reader data loaded. Call load_experienced_reader_data() first.")
            return

        print("\n" + "=" * 80)
        print("EXPERIENCED READER SCORES (from -es2 and -es3 files)")
        print("  Settings 0-2 (-es2 files):")
        print("    Setting 0: predict culprit (general instructions only, 0-shot)")
        print("    Setting 1: 9 previous stories from same model given")
        print("    Setting 2: 3 previous stories from same model given")
        print("  Settings 3-5 (-es3 files):")
        print("    Setting 3: different prompt, no previous stories (0-shot alt)")
        print("    Setting 4: 2 previous stories from same model given")
        print("    Setting 5: 5 previous stories from same model given")
        print("=" * 80)

        for setting in [0, 1, 2, 3, 4, 5]:
            setting_label = EXPERIENCED_READER_LABELS[setting]
            print(f"\n{'-' * 70}")
            print(f"{setting_label}")
            print(f"{'-' * 70}")

            scores_by_cat = {'artificial': ([], [], []), 'real': ([], [], [])}

            # Determine which data store to iterate
            data_store = self.experienced_reader_data if setting <= 2 else self.experienced_reader_data_es3

            for key, value in data_store.items():
                model_name, story_type, story_id = key

                exp_dist, true_idx = self.get_experienced_reader_distributions(
                    model_name, story_type, story_id, setting=setting
                )
                if exp_dist is None or true_idx is None:
                    continue

                llm_key = (model_name, story_type, story_id)
                if llm_key not in self.llm_data:
                    continue

                story_label = story_type if story_type in ('poirot', 'sherlock') else model_name
                naive_dist, naive_true_idx = self.get_naive_reader_probs(f"{story_label} {story_id}")
                sampling_dist, _, _, _, _ = self.get_llm_distributions(model_name, story_type, story_id)
                if naive_dist is None or naive_true_idx is None or sampling_dist is None:
                    continue
                if len(naive_dist) != len(exp_dist):
                    continue

                exp_true = exp_dist[:, true_idx]
                naive_true = naive_dist[:, true_idx]

                fp = np.mean(exp_true - naive_true) * 25
                surprise = np.mean(naive_true) * 25
                coherence = np.mean(exp_true) * 25

                cat = 'real' if story_type in ('poirot', 'sherlock') else 'artificial'
                scores_by_cat[cat][0].append(fp)
                scores_by_cat[cat][1].append(surprise)
                scores_by_cat[cat][2].append(coherence)

            for cat_label, cat_key in [("ARTIFICIAL STORIES", "artificial"), ("REAL STORIES", "real")]:
                fp_scores, surprise_scores, coherence_scores = scores_by_cat[cat_key]
                print(f"\n  {cat_label}:")
                if fp_scores:
                    print(f"    n stories = {len(fp_scores)}")
                    print(f"    Fair Play (x25):  Mean={np.mean(fp_scores):.3f}, Std={np.std(fp_scores):.3f}")
                    print(f"    Fair Play:        Mean={np.mean(fp_scores) / 25:.3f}")
                    print(f"    Surprise (x25):   Mean={np.mean(surprise_scores):.3f}, Std={np.std(surprise_scores):.3f}")
                    print(f"    Surprise:         Mean={np.mean(surprise_scores) / 25:.3f}")
                    print(f"    Coherence (x25):  Mean={np.mean(coherence_scores):.3f}, Std={np.std(coherence_scores):.3f}")
                    print(f"    Coherence:        Mean={np.mean(coherence_scores) / 25:.3f}")
                else:
                    print(f"    No valid stories for this setting.")

        print("\n" + "=" * 80)
        print("END OF EXPERIENCED READER SCORES")
        print("=" * 80 + "\n")

    def print_participant_experience(self):
        """Report self-reported familiarity with detective stories per participant.

        Extracts the familiarity rating (1–5) from each form submission, deduplicates
        by user_id (taking the mean if a participant appears more than once), and prints
        summary statistics and the full distribution.

        Corresponds to TODO: "Report self-reported experience with detective stories."
        """
        import pandas as pd

        # Collect (user_id, familiarity) from all loaded story sheets
        records = {}  # uid -> list of familiarity scores
        response_counts = {}  # uid -> number of stories responded to
        for story_data in self.human_data.values():
            df = story_data['df']
            if len(df) == 0:
                continue

            user_col = df.columns[1]  # 'Personal ID for this task'
            fam_cols = [c for c in df.columns if 'familiarity' in c.lower()]
            if not fam_cols:
                continue
            fam_col = fam_cols[0]

            for _, row in df.iterrows():
                uid = str(row[user_col]).strip()
                fam = row[fam_col]
                if pd.notna(fam) and uid:
                    records.setdefault(uid, []).append(float(fam))
                    response_counts[uid] = response_counts.get(uid, 0) + 1

        if not records:
            print("No familiarity data found.")
            return

        # Average per participant (should be the same value each time, but average to be safe)
        per_participant = {uid: np.mean(scores) for uid, scores in records.items()}
        scores = np.array(list(per_participant.values()))

        from collections import Counter
        dist = Counter(round(s) for s in scores)

        print("\n" + "=" * 60)
        print("PARTICIPANT SELF-REPORTED FAMILIARITY WITH DETECTIVE STORIES")
        print("  Scale: 1 (not familiar) – 5 (highly familiar)")
        print("=" * 60)
        n_responses = np.array(list(response_counts.values()))

        print(f"  N participants : {len(scores)}")
        print(f"  Mean ± SD      : {scores.mean():.2f} ± {scores.std():.2f}")
        print(f"  Median         : {np.median(scores):.1f}")
        print(f"  Range          : {scores.min():.0f} – {scores.max():.0f}")
        print(f"  Distribution:")
        for k in sorted(dist):
            bar = "#" * dist[k]
            print(f"    {k}: {dist[k]:2d}  {bar}")
        print(f"\n  Responses per participant:")
        print(f"    Mean ± SD : {n_responses.mean():.2f} ± {n_responses.std():.2f}")
        print(f"    Median    : {np.median(n_responses):.1f}")
        print(f"    Range     : {n_responses.min()} – {n_responses.max()}")

    def print_reading_curve_stats(self):
        """Print FPUB and human FP scores per story, for inserting into LaTeX subcaptions.

        Corresponds to TODO: "Add FPUB and human FP scores in each reading curve subcaption."
        """
        print("\n" + "=" * 70)
        print("READING CURVE STATS  (FPUB and Human FP per story)")
        print("  FPUB    = know-it-all fair play  (mean sampling - naive)")
        print("  Human FP = human fair play        (mean human - naive)")
        print("=" * 70)

        for story_name in sorted(self.human_data.keys()):
            human_dist, _, _ = self.get_human_distribution(story_name, use_argmax=False)
            if human_dist is None:
                continue

            naive_dist, true_idx = self.get_naive_reader_probs(story_name, use_argmax=False)
            if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                continue

            model_key, story_id = story_name.split()
            is_real = model_key in ['poirot', 'sherlock']

            if is_real:
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    REAL_STORY_S_MODEL, model_key, story_id, use_argmax=False)
            else:
                model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    model_key, model_key2, story_id, use_argmax=False)

            naive_true = naive_dist[:, true_idx]
            human_fp = float(np.mean(human_dist[:, true_idx] - naive_true))

            fpub_str = "N/A"
            if sampling_dist is not None and len(sampling_dist) == len(human_dist):
                fpub = float(np.mean(sampling_dist[:, true_idx] - naive_true))
                fpub_str = f"{fpub:+.3f}"

            print(f"  {story_name:<40s}  FPUB={fpub_str}  Human FP={human_fp:+.3f}")

    def compute_name_diversity(self, stories_dir=None):
        """Compute name diversity (entropy per role) from _w_details2.json files.

        For each model and each character role (culprit, victim, detective), loads the
        annotated story files and computes the Shannon entropy of the first-name
        distribution across stories.  High entropy = diverse names; low entropy = repetitive.

        Corresponds to TODO: "Add a diversity metric (e.g., entropy of most-common-name
        distribution per role). Need to calculate from the data."

        Args:
            stories_dir: Path to the stories/ directory.  Defaults to ../stories/
                         relative to this file.
        """
        import json
        import math
        from pathlib import Path
        from collections import Counter

        if stories_dir is None:
            stories_dir = Path(__file__).parent.parent / "stories"
        stories_dir = Path(stories_dir)

        # Map analysis model keys → _w_details2.json filenames
        model_to_file = {
            'gpt-4o':                 'gpt-4o_w_suspects_sbs_d25_w_details2.json',
            'gpt-4o-mini':            'gpt-4o-mini_w_suspects_sbs_d25_w_details2.json',
            'gemini-2.5-flash-tb0':   'gemini-2.5-flash_tb0_w_suspects_sbs_d25_w_details2.json',
            'gemini-2.5-flash-tb-1':  'gemini-2.5-flash_tb-1_w_suspects_sbs_d25_w_details2.json',
            'gemini-2.5-pro':         'gemini-2.5-pro_tb-1_w_suspects_sbs_d25_w_details2.json',
            'Llama-3.3-70B-Instruct': 'Llama-3.3-70B-Instruct_w_suspects_sbs_d25_w_details2.json',
            'gemini-3-flash-preview':  'gemini-3-flash-preview_w_suspects_sbs_d25_w_details2.json',
            'gemini-3.1-pro-preview':  'gemini-3.1-pro-preview_w_suspects_sbs_d25_w_details2.json',
            'poirot':                  'poirot_w_suspects_w_details2.json',
            'sherlock':                'sherlock_w_suspects_w_details2.json',
        }

        roles = ['culprit', 'victim', 'detective']

        print("\n" + "=" * 70)
        print("NAME DIVERSITY  (Shannon entropy of first-name distribution per role)")
        print("  entropy=0 → same name every story; higher → more diverse")
        print("=" * 70)

        all_results = {}

        for model, filename in model_to_file.items():
            path = stories_dir / filename
            if not path.exists():
                continue

            with open(path) as f:
                stories = json.load(f)
            if isinstance(stories, dict):
                stories = list(stories.values())

            print(f"\n{model}  (n={len(stories)}):")
            model_results = {}

            for role in roles:
                field = f'{role}_first_name'
                names = [s.get(field, '').strip().lower() for s in stories
                         if s.get(field, '').strip()]
                if not names:
                    print(f"  {role:<12s} — no data")
                    continue

                counts = Counter(names)
                total = sum(counts.values())
                probs = [c / total for c in counts.values()]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
                top_name, top_freq = counts.most_common(1)[0]

                model_results[role] = {
                    'entropy': entropy,
                    'top_name': top_name,
                    'top_freq': top_freq,
                    'n': total,
                    'n_unique': len(counts),
                }
                print(f"  {role:<12s}  entropy={entropy:.2f} bits  "
                      f"unique={len(counts)}/{total}  "
                      f"top='{top_name}' ({top_freq/total:.0%})")

            all_results[model] = model_results

        return all_results

    def print_early_reveal_stats(self, paragraphs=None, threshold=None):
        """Report how often the know-it-all reader assigns P(true culprit) > threshold
        at early checkpoints (raw probabilities, not argmax).

        Statistical test: H0 = know-it-all readers guess uniformly among n_suspects.
        Per story, null probability = P(Bin(nsamp, 1/n_suspects) > threshold * nsamp).
        P-value = P(Poisson(E) >= k) where E = sum of per-story null probabilities.

        Args:
            paragraphs: iterable of paragraph numbers (from ids) to check.
                        Defaults to EARLY_REVEAL_PARAGRAPHS from config.
            threshold:  probability threshold.  Defaults to EARLY_REVEAL_THRESHOLD.
        """
        from scipy.stats import binom as _binom, poisson as _poisson

        if paragraphs is None:
            paragraphs = EARLY_REVEAL_PARAGRAPHS
        if threshold is None:
            threshold = EARLY_REVEAL_THRESHOLD

        paragraphs = list(paragraphs)

        print("\n" + "=" * 72)
        print(f"EARLY REVEAL ANALYSIS  (know-it-all P(true culprit) > {threshold})")
        print(f"  Checkpoints: paragraphs {paragraphs}  (raw probabilities, not argmax)")
        print(f"  H0: readers guess uniformly; p-value via Poisson(E_null) test")
        print("=" * 72)

        # Prefer nsamp=20 over nsamp=5; deduplicate per story
        story_best = {}
        for key, value in self.llm_data.items():
            mn, st, sid = key
            nsamp = int(value['params'].get('nsamp', '5'))
            is_real = st in ('poirot', 'sherlock')
            group = st if is_real else mn
            sk = f"{group}_{sid}"
            if sk not in story_best or nsamp == 20:
                story_best[sk] = (key, group, is_real)

        # counts[group][para]    = [wins, total]
        # null_exp[group][para]  = sum of per-story null probabilities (expected wins under H0)
        counts   = {}
        null_exp = {}
        overall  = {p: [0, 0, 0.0] for p in paragraphs}  # [wins, total, null_E]

        for sk, (key, group, is_real) in story_best.items():
            mn, st, sid = key
            value        = self.llm_data[key]
            data         = value['data']
            nsamp        = int(value['params'].get('nsamp', '5'))
            ids          = data.get('ids', [])
            sampling_raw = data.get('sampling', [])
            real_probs   = data.get('real_probs', [])

            if not sampling_raw or not ids:
                continue

            sampling = np.array(sampling_raw, dtype=float)
            n_steps, n_suspects = sampling.shape

            if real_probs:
                true_idx = int(np.argmax(real_probs))
            else:
                true_idx = int(np.argmax(sampling[-1]))

            if group not in counts:
                counts[group]   = {p: [0, 0]  for p in paragraphs}
                null_exp[group] = {p:  0.0     for p in paragraphs}

            for para in paragraphs:
                if para not in ids:
                    continue
                step = ids.index(para)
                if step >= n_steps:
                    continue

                # Null probability for this story: P(Bin(nsamp, 1/n_suspects) > threshold*nsamp)
                null_p = _binom.sf(int(threshold * nsamp), nsamp, 1.0 / n_suspects)

                counts[group][para][1]  += 1
                null_exp[group][para]   += null_p
                overall[para][1]        += 1
                overall[para][2]        += null_p

                if sampling[step, true_idx] > threshold:
                    counts[group][para][0] += 1
                    overall[para][0]       += 1

        # ---- output ----
        col_w = 20
        pad   = 28

        def _fmt(wins, total, exp):
            if total == 0:
                return "n/a", ""
            pct   = wins / total * 100
            count = f"{wins}/{total} ({pct:.0f}%)"
            if wins == 0:
                p_val = 1.0
            else:
                p_val = float(_poisson.sf(wins - 1, max(exp, 1e-12)))
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            pval  = f"p={p_val:.3f}{sig}"
            return count, pval

        header = f"  {'Model/type':<{pad}}" + "".join(f"{'Para '+str(p):>{col_w}}" for p in paragraphs)
        print(header)
        print("  " + "-" * pad + "-" * col_w * len(paragraphs))

        real_groups  = sorted(g for g in counts if g in ('poirot', 'sherlock'))
        model_groups = sorted((g for g in counts if g not in ('poirot', 'sherlock')),
                              key=lambda m: display_names.get(m, m))

        for group in real_groups + model_groups:
            label    = display_names.get(group, group)
            row_cnt  = f"  {label:<{pad}}"
            row_pval = f"  {'':{ pad}}"
            for para in paragraphs:
                wins, total = counts[group][para]
                exp         = null_exp[group][para]
                cnt, pval   = _fmt(wins, total, exp)
                row_cnt  += f"{cnt:>{col_w}}"
                row_pval += f"{pval:>{col_w}}"
            print(row_cnt)
            print(row_pval)

        print("  " + "-" * pad + "-" * col_w * len(paragraphs))
        row_cnt  = f"  {'OVERALL':<{pad}}"
        row_pval = f"  {'':{ pad}}"
        for para in paragraphs:
            wins, total, exp = overall[para]
            cnt, pval        = _fmt(wins, total, exp)
            row_cnt  += f"{cnt:>{col_w}}"
            row_pval += f"{pval:>{col_w}}"
        print(row_cnt)
        print(row_pval)

    def _compute_model_metrics(self, naive_mode=None):
        """Compute per-model quality metrics for artificial stories.

        Returns:
            model_metrics – dict {model_name: {surprise, coherence, fp_gt1_rate,
                            human_fp_gt1_rate, cl_def_gt1_rate, no_early_reveal_p1, n}}
            ordered_pairs – list of (better, worse) model name tuples
            _META         – model metadata dict
            _cmp          – comparison function
        """
        _META = {
            'gemini-1.5-flash':       ('gemini', 1.5, 0, 0),
            'gemini-2.5-flash-tb0':   ('gemini', 2.5, 0, 0),  # subtier 0
            'gemini-2.5-flash-tb-1':  ('gemini', 2.5, 0, 1),  # subtier 1: TB-1 > TB-0
            'gemini-3-flash-preview':  ('gemini', 3.0, 0, 0),
            'gemini-1.5-pro':          ('gemini', 1.5, 1, 0),
            'gemini-2.5-pro':          ('gemini', 2.5, 1, 0),
            'gemini-3.1-pro-preview':  ('gemini', 3.1, 1, 0),
            'Llama-3.1-70B-Instruct':  ('llama',  3.1, 70, 0),
            'Llama-3.3-70B-Instruct':  ('llama',  3.3, 70, 0),
            'Llama-3.1-8B-Instruct':   ('llama',  3.1,  8, 0),
            'Llama-3.2-3B-Instruct':   ('llama',  3.2,  3, 0),
            'Llama-3.2-1B-Instruct':   ('llama',  3.2,  1, 0),
            'gpt-4o':                  ('gpt_4o', 4.0,  1, 0),
            'gpt-4o-mini':             ('gpt_4o', 4.0,  0, 0),
        }

        def _cmp(ma, mb):
            fa, va, sa = ma[0], ma[1], ma[2]
            sub_a = ma[3] if len(ma) > 3 else 0
            fb, vb, sb = mb[0], mb[1], mb[2]
            sub_b = mb[3] if len(mb) > 3 else 0
            if fa != fb:
                return 0
            if fa == 'gemini':
                maj_a, maj_b = int(va), int(vb)
                tier_a, tier_b = sa, sb
                if maj_a == maj_b:
                    if tier_a != tier_b:
                        return +1 if tier_a > tier_b else -1
                    if sub_a != sub_b:
                        return +1 if sub_a > sub_b else -1
                    return +1 if va > vb else (-1 if vb > va else 0)
                else:
                    if tier_a != tier_b:
                        return 0
                    return +1 if maj_a > maj_b else -1
            newer  = (va > vb) - (vb > va)
            larger = (sa > sb) - (sb > sa)
            if newer == 0:  return larger
            if larger == 0 or newer == larger: return newer
            return 0

        saved_mode = self.naive_reader_mode
        if naive_mode is not None:
            self.naive_reader_mode = naive_mode

        story_best = {}
        for key, value in self.llm_data.items():
            mn, st, sid = key
            if st in ('poirot', 'sherlock'):
                continue
            nsamp = int(value['params'].get('nsamp', '5'))
            sk = f"{mn}_{sid}"
            if sk not in story_best or nsamp == 20:
                story_best[sk] = (key, mn)

        n_total_per_model = {}
        for sk, (key, mn) in story_best.items():
            n_total_per_model[mn] = n_total_per_model.get(mn, 0) + 1

        model_records = {}
        for sk, (key, mn) in story_best.items():
            _, st, sid = key
            sampling_dist, _, _, _, _ = self.get_llm_distributions(mn, st, sid)
            naive_dist, true_idx = self.get_naive_reader_probs(f"{mn} {sid}")
            if sampling_dist is None or naive_dist is None or true_idx is None:
                continue
            if len(sampling_dist) != len(naive_dist):
                continue

            naive_true    = naive_dist[:, true_idx]
            sampling_true = sampling_dist[:, true_idx]
            surprise  = np.mean(naive_true) * 25
            coherence = np.mean(sampling_true) * 25
            fp_llm    = coherence - surprise

            loop_mode = self.naive_reader_mode
            self.naive_reader_mode = "default"
            nd_def, ti_def = self.get_naive_reader_probs(f"{mn} {sid}")
            self.naive_reader_mode = "clueless"
            nd_cl,  ti_cl  = self.get_naive_reader_probs(f"{mn} {sid}")
            self.naive_reader_mode = loop_mode
            cl_def_diff = None
            if nd_def is not None and nd_cl is not None and ti_def is not None and ti_cl is not None:
                cl_def_diff = (np.mean(nd_cl[:, ti_cl]) - np.mean(nd_def[:, ti_def])) * 25

            hg = mn if 'gemini-2.5-flash' not in mn else 'gemini-2.5-flash'
            human_dist, _, _ = self.get_human_distribution(f"{hg} {sid}")
            fp_human = None
            if human_dist is not None and len(human_dist) == len(naive_dist):
                fp_human = (np.mean(human_dist[:, true_idx]) - np.mean(naive_true)) * 25

            # Early reveal at paragraph 1
            raw_val = self.llm_data[key]
            data_d  = raw_val['data']
            ids_d   = data_d.get('ids', [])
            samp_d  = data_d.get('sampling', [])
            early_reveal_p1 = None
            if samp_d and ids_d and 1 in ids_d:
                step  = ids_d.index(1)
                s_arr = np.array(samp_d, dtype=float)
                if step < len(s_arr):
                    rp    = data_d.get('real_probs', [])
                    t_idx = int(np.argmax(rp)) if rp else int(np.argmax(s_arr[-1]))
                    early_reveal_p1 = float(s_arr[step, t_idx] > EARLY_REVEAL_THRESHOLD)

            model_records.setdefault(mn, []).append({
                'surprise':        surprise,
                'coherence':       coherence,
                'fp_llm':          fp_llm,
                'fp_human':        fp_human,
                'cl_def_diff':     cl_def_diff,
                'early_reveal_p1': early_reveal_p1,
            })

        model_metrics = {}
        for mn, recs in model_records.items():
            fp_h  = [r['fp_human']    for r in recs if r['fp_human']    is not None]
            cl_ds = [r['cl_def_diff'] for r in recs if r['cl_def_diff'] is not None]
            er    = [r['early_reveal_p1'] for r in recs if r['early_reveal_p1'] is not None]
            n_valid = len(recs)
            n_total = n_total_per_model.get(mn, n_valid)
            model_metrics[mn] = {
                'surprise':           np.mean([r['surprise']  for r in recs]),
                'coherence':          np.mean([r['coherence'] for r in recs]),
                'fp_gt1_rate':        np.mean([r['fp_llm']    > 1 for r in recs]),
                'human_fp_gt1_rate':  np.mean([v > 1 for v in fp_h])  if fp_h  else None,
                'human_fp_mean':      np.mean(fp_h)                    if fp_h  else None,
                'cl_def_gt1_rate':    np.mean([v > 1 for v in cl_ds]) if cl_ds else None,
                'cl_def_mean':        np.mean(cl_ds)                   if cl_ds else None,
                'no_early_reveal_p1': 1.0 - np.mean(er)               if er    else None,
                'gen_success':        MODEL_GEN_SUCCESS.get(mn),
                'n': n_valid,
                'n_total': n_total,
            }

        all_models = list(model_metrics.keys())
        ordered_pairs = []
        for i, ma in enumerate(all_models):
            for mb in all_models[i + 1:]:
                meta_a = _META.get(ma)
                meta_b = _META.get(mb)
                if meta_a is None or meta_b is None:
                    continue
                rel = _cmp(meta_a, meta_b)
                if rel == +1:
                    ordered_pairs.append((ma, mb))
                elif rel == -1:
                    ordered_pairs.append((mb, ma))

        self.naive_reader_mode = saved_mode
        return model_metrics, ordered_pairs, _META, _cmp

    def analyze_model_ordering(self, naive_mode=None):
        """Test whether 'newer/larger = better' holds for five quality metrics.

        Defines a partial ordering between models:
          - Same line (flash/pro/llama/gpt), newer generation > older
            (valid when size is the same or unknown)
          - Same line, same generation, larger size > smaller size
          - Conflicting rules (newer but smaller) → no ordering for that pair
          - Different families (gemini vs llama etc.) → no ordering

        For each ordered pair (better, worse) a sign test checks whether the
        'better' model scores higher, reported one-sided (H1: P > 0.5).

        Metrics tested (per-model means over artificial stories):
            surprise          – mean naive P(true culprit) × 25
            coherence_llm     – mean sampling P(true culprit) × 25
            fp_gt1_rate       – fraction of stories with LLM FP > 1
            human_fp_gt1_rate – fraction of stories with human FP > 1
            cl_def_gt1_rate   – fraction of stories with (clueless − default) × 25 > 1

        Args:
            naive_mode: naive reader mode to use for surprise / coherence / FP.
                        If None, uses the current self.naive_reader_mode.
        """
        import re
        from scipy import stats as _stats

        saved_mode = self.naive_reader_mode
        if naive_mode is not None:
            self.naive_reader_mode = naive_mode

        # ------------------------------------------------------------------ #
        # Model metadata: (family, version, tier, subtier)                  #
        # gemini: tier 0=flash, 1=pro; subtier 1=TB-1 > 0=TB-0             #
        # llama:  size in B                                                  #
        # gpt_4o: 0=mini, 1=full                                             #
        # ------------------------------------------------------------------ #
        _META = {
            'gemini-1.5-flash':       ('gemini', 1.5, 0, 0),
            'gemini-2.5-flash-tb0':   ('gemini', 2.5, 0, 0),  # subtier 0
            'gemini-2.5-flash-tb-1':  ('gemini', 2.5, 0, 1),  # subtier 1: TB-1 > TB-0
            'gemini-3-flash-preview':  ('gemini', 3.0, 0, 0),
            'gemini-1.5-pro':          ('gemini', 1.5, 1, 0),
            'gemini-2.5-pro':          ('gemini', 2.5, 1, 0),
            'gemini-3.1-pro-preview':  ('gemini', 3.1, 1, 0),
            'Llama-3.1-70B-Instruct':  ('llama',  3.1, 70, 0),
            'Llama-3.3-70B-Instruct':  ('llama',  3.3, 70, 0),
            'Llama-3.1-8B-Instruct':   ('llama',  3.1,  8, 0),
            'Llama-3.2-3B-Instruct':   ('llama',  3.2,  3, 0),
            'Llama-3.2-1B-Instruct':   ('llama',  3.2,  1, 0),
            'gpt-4o':                  ('gpt_4o', 4.0,  1, 0),
            'gpt-4o-mini':             ('gpt_4o', 4.0,  0, 0),
        }

        def _cmp(ma, mb):
            """Return +1 if A better, -1 if B better, 0 if no ordering."""
            fa, va, sa = ma[0], ma[1], ma[2]
            sub_a = ma[3] if len(ma) > 3 else 0
            fb, vb, sb = mb[0], mb[1], mb[2]
            sub_b = mb[3] if len(mb) > 3 else 0
            if fa != fb:
                return 0
            if fa == 'gemini':
                # tier: 0=flash, 1=pro.  pro > flash at same major version;
                # same tier: check subtier (TB-1 > TB-0), then newer major wins;
                # cross-major cross-tier: no ordering.
                maj_a, maj_b = int(va), int(vb)
                tier_a, tier_b = sa, sb
                if maj_a == maj_b:
                    if tier_a != tier_b:
                        return +1 if tier_a > tier_b else -1
                    if sub_a != sub_b:
                        return +1 if sub_a > sub_b else -1
                    return +1 if va > vb else (-1 if vb > va else 0)
                else:
                    if tier_a != tier_b:
                        return 0   # conflicting: newer but lower tier
                    return +1 if maj_a > maj_b else -1
            newer = (va > vb) - (vb > va)
            larger = (sa > sb) - (sb > sa)
            if newer == 0:
                return larger
            if larger == 0 or newer == larger:
                return newer
            return 0

        # ------------------------------------------------------------------ #
        # Collect per-story data for artificial stories                       #
        # ------------------------------------------------------------------ #
        story_best = {}                        # story_key -> (key, model_name)
        for key, value in self.llm_data.items():
            mn, st, sid = key
            if st in ['poirot', 'sherlock']:
                continue
            nsamp = int(value['params'].get('nsamp', '5'))
            sk = f"{mn}_{sid}"
            if sk not in story_best or nsamp == 20:
                story_best[sk] = (key, mn)

        model_records = {}                     # model_name -> list of metric dicts
        for sk, (key, mn) in story_best.items():
            _, st, sid = key

            sampling_dist, _, _, _, _ = self.get_llm_distributions(mn, st, sid)
            naive_dist, true_idx = self.get_naive_reader_probs(f"{mn} {sid}")
            if sampling_dist is None or naive_dist is None or true_idx is None:
                continue
            if len(sampling_dist) != len(naive_dist):
                continue

            naive_true    = naive_dist[:, true_idx]
            sampling_true = sampling_dist[:, true_idx]
            surprise   = np.mean(naive_true) * 25
            coherence  = np.mean(sampling_true) * 25
            fp_llm     = coherence - surprise

            # Clueless − default diff (mode-independent)
            loop_mode = self.naive_reader_mode
            self.naive_reader_mode = "default"
            nd_def, ti_def = self.get_naive_reader_probs(f"{mn} {sid}")
            self.naive_reader_mode = "clueless"
            nd_cl,  ti_cl  = self.get_naive_reader_probs(f"{mn} {sid}")
            self.naive_reader_mode = loop_mode
            cl_def_diff = None
            if nd_def is not None and nd_cl is not None and ti_def is not None and ti_cl is not None:
                cl_def_diff = (np.mean(nd_cl[:, ti_cl]) - np.mean(nd_def[:, ti_def])) * 25

            # Human FP
            hg = mn if 'gemini-2.5-flash' not in mn else 'gemini-2.5-flash'
            human_dist, _, _ = self.get_human_distribution(f"{hg} {sid}")
            fp_human = None
            if human_dist is not None and len(human_dist) == len(naive_dist):
                fp_human = (np.mean(human_dist[:, true_idx]) - np.mean(naive_true)) * 25

            model_records.setdefault(mn, []).append({
                'surprise':    surprise,
                'coherence':   coherence,
                'fp_llm':      fp_llm,
                'fp_human':    fp_human,
                'cl_def_diff': cl_def_diff,
            })

        # Aggregate per model
        model_metrics = {}
        for mn, recs in model_records.items():
            fp_h    = [r['fp_human']    for r in recs if r['fp_human']    is not None]
            cl_difs = [r['cl_def_diff'] for r in recs if r['cl_def_diff'] is not None]
            model_metrics[mn] = {
                'surprise':          np.mean([r['surprise']  for r in recs]),
                'coherence':         np.mean([r['coherence'] for r in recs]),
                'fp_gt1_rate':       np.mean([r['fp_llm']    > 1 for r in recs]),
                'human_fp_gt1_rate': np.mean([v > 1 for v in fp_h])    if fp_h    else None,
                'cl_def_gt1_rate':   np.mean([v > 1 for v in cl_difs]) if cl_difs else None,
                'n': len(recs),
            }

        # ------------------------------------------------------------------ #
        # Build ordered pairs                                                 #
        # ------------------------------------------------------------------ #
        all_models = list(model_metrics.keys())
        ordered_pairs = []
        for i, ma in enumerate(all_models):
            for mb in all_models[i + 1:]:
                meta_a = _META.get(ma)
                meta_b = _META.get(mb)
                if meta_a is None or meta_b is None:
                    continue
                rel = _cmp(meta_a, meta_b)
                if rel == +1:
                    ordered_pairs.append((ma, mb))
                elif rel == -1:
                    ordered_pairs.append((mb, ma))

        # ------------------------------------------------------------------ #
        # Sign test for each metric                                           #
        # ------------------------------------------------------------------ #
        print("\n" + "=" * 72)
        print("MODEL ORDERING TEST  (newer / larger = better hypothesis)")
        print(f"  Naive mode: {self.naive_reader_mode}")
        print("=" * 72)

        print(f"\nOrdered pairs (better > worse):  n = {len(ordered_pairs)}")
        for a, b in ordered_pairs:
            print(f"  {display_names.get(a, a):<30}  >  {display_names.get(b, b)}")

        METRICS = [
            ('surprise',          'Surprise (×25)'),
            ('coherence',         'Coherence LLM (×25)'),
            ('fp_gt1_rate',       'LLM FP > 1 rate'),
            ('human_fp_gt1_rate', 'Human FP > 1 rate'),
            ('cl_def_gt1_rate',   'Clueless−Default > 1 rate'),
        ]

        print(f"\n  Sign test (one-sided H1: better model scores higher):")
        print(f"  {'Metric':<30}  {'n':>4}  {'wins':>5}  {'pct':>6}  {'p-value':>9}")
        print(f"  {'-'*30}  {'-'*4}  {'-'*5}  {'-'*6}  {'-'*9}")

        results = {}
        for mkey, mlabel in METRICS:
            wins = losses = ties = 0
            for better, worse in ordered_pairs:
                va = model_metrics.get(better, {}).get(mkey)
                vb = model_metrics.get(worse,  {}).get(mkey)
                if va is None or vb is None:
                    continue
                if   va > vb: wins   += 1
                elif vb > va: losses += 1
                else:         ties   += 1

            n_valid = wins + losses
            if n_valid == 0:
                print(f"  {mlabel:<30}  {'n/a':>4}  {'n/a':>5}  {'n/a':>6}  {'n/a':>9}")
                results[mkey] = None
                continue

            pct = wins / n_valid * 100
            try:
                p_val = _stats.binomtest(wins, n_valid, p=0.5, alternative='greater').pvalue
            except AttributeError:
                p_val = _stats.binom_test(wins, n_valid, p=0.5, alternative='greater')
            sig = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else ''
            print(f"  {mlabel:<30}  {n_valid:>4}  {wins:>5}  {pct:5.1f}%  {p_val:>8.4f}{sig}")
            results[mkey] = {'wins': wins, 'losses': losses, 'ties': ties, 'p': p_val}

        # Per-model metric table
        print(f"\n  Per-model metrics:")
        header = f"  {'Model':<30}  {'n':>4}  {'Surp':>6}  {'Coh':>6}  {'FP>1':>6}  {'hFP>1':>6}  {'Cl>1':>6}"
        print(header)
        print(f"  {'-'*30}  {'-'*4}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}")
        for mn in sorted(model_metrics.keys(), key=lambda m: display_names.get(m, m)):
            mm = model_metrics[mn]
            hfp = f"{mm['human_fp_gt1_rate']:6.3f}" if mm['human_fp_gt1_rate'] is not None else "   n/a"
            cl  = f"{mm['cl_def_gt1_rate']:6.3f}"   if mm['cl_def_gt1_rate']   is not None else "   n/a"
            print(f"  {display_names.get(mn, mn):<30}  {mm['n']:>4}  "
                  f"{mm['surprise']:6.2f}  {mm['coherence']:6.2f}  "
                  f"{mm['fp_gt1_rate']:6.3f}  {hfp}  {cl}")

        self.naive_reader_mode = saved_mode
        return results

    def compare_revelation_points(self, stories_dir=None):
        """Compare LLM-annotated revelation points (true_culprit_paragraph from _w_details files)
        with the threshold-based revelation point derived from the naive reader distribution
        (first paragraph where the naive reader's probability for the true culprit crosses
        NAIVE_THRESHOLD and stays above it).

        Prints a per-story table and summary statistics (mean diff, MAE, Pearson r).
        Returns a list of dicts with the per-story results.
        """
        from pathlib import Path
        from scipy import stats as scipy_stats

        if stories_dir is None:
            stories_dir = Path(self.json_dir).parent / "stories"
        stories_dir = Path(stories_dir)

        real_types = {'poirot', 'sherlock', 'rivals', 'edwin'}

        # Build a lookup: (model_or_type, suspects_tuple) -> llm_data key
        # For artificial stories: model_or_type = model_name (key[0])
        # For real stories: model_or_type = story_type (key[1], e.g. 'poirot'),
        #   and only REAL_STORY_S_MODEL entries are used (consistent with the rest of the analysis).
        suspects_to_key = {}
        for key, value in self.llm_data.items():
            mn, st, sid = key
            suspects = value['data'].get('suspects')
            if not suspects:
                continue
            is_real = st in real_types
            if is_real and mn != REAL_STORY_S_MODEL:
                continue  # only index real stories under the canonical sampling model
            lookup_model = st if is_real else mn
            suspects_to_key[(lookup_model, tuple(suspects))] = key

        rows = []

        # Prefer _w_details2.json over _w_details.json when both exist for the same model
        best_files = {}
        for f in stories_dir.glob('*_w_details*.json'):
            base = f.stem.replace('_w_details2', '').replace('_w_details', '')
            if base not in best_files or '_w_details2' in f.stem:
                best_files[base] = f

        for details_file in sorted(best_files.values()):
            stem = details_file.stem  # e.g. "gemini-1.5-flash_w_suspects_sbs_d25_w_details2"
            base = stem.replace('_w_details2', '').replace('_w_details', '')
            model_part = base.split('_w_suspects')[0]
            model_name = model_part.replace('_tb', '-tb')  # gemini-2.5-flash_tb0 -> gemini-2.5-flash-tb0

            with open(details_file) as f:
                stories = json.load(f)
            if isinstance(stories, dict):
                stories = list(stories.values())

            for story in stories:
                llm_revelation = story.get('true_culprit_paragraph')
                suspects = story.get('suspects')
                if llm_revelation is None or not suspects:
                    continue
                intro_end = story.get('last_introduction_paragraph')

                is_real = model_name in real_types
                lookup_model = model_name  # for real: story_type; for artificial: model_name
                key = suspects_to_key.get((lookup_model, tuple(suspects)))
                if key is None:
                    continue

                data = self.llm_data[key]['data']
                direct = data.get('direct', {})
                ids = data.get('ids', [])
                real_probs = data.get('real_probs', [])

                if '1' not in direct or not ids:
                    continue

                naive_dist = np.array(direct['1'])
                if len(naive_dist) == 0:
                    continue

                if real_probs:
                    true_idx = int(np.argmax(real_probs))
                else:
                    true_idx = int(np.argmax(naive_dist[-1]))

                # Find first step where naive prob >= NAIVE_THRESHOLD and stays there
                true_probs = naive_dist[:, true_idx]
                threshold_step = len(naive_dist) - 1
                for i in range(len(true_probs)):
                    if true_probs[i] >= NAIVE_THRESHOLD:
                        if all(true_probs[j] >= NAIVE_THRESHOLD for j in range(i, len(true_probs))):
                            threshold_step = i
                            break

                threshold_paragraph = ids[threshold_step]
                diff = int(threshold_paragraph) - int(llm_revelation)

                rows.append({
                    'model': model_name,
                    'story_id': key[2],
                    'llm_revelation': int(llm_revelation),
                    'threshold_revelation': int(threshold_paragraph),
                    'diff': diff,
                    'abs_diff': abs(diff),
                    'intro_end': int(intro_end) if intro_end not in (None, '') else None,
                })

        if not rows:
            print("No matching stories found.")
            return rows

        print("\n" + "=" * 70)
        print("REVELATION POINT COMPARISON")
        print(f"  Threshold method: naive prob >= {NAIVE_THRESHOLD} and stays above")
        print(f"  LLM method: true_culprit_paragraph from _w_details files")
        print("=" * 70)

        # Per-model summary
        models = sorted(set(r['model'] for r in rows))
        for model in models:
            model_rows = [r for r in rows if r['model'] == model]
            llm_vals = [r['llm_revelation'] for r in model_rows]
            thr_vals = [r['threshold_revelation'] for r in model_rows]
            diffs = [r['diff'] for r in model_rows]
            abs_diffs = [r['abs_diff'] for r in model_rows]

            print(f"\n{model}  (n={len(model_rows)}):")
            print(f"  {'Story':>8}  {'IntroEnd':>9}  {'LLM':>6}  {'Threshold':>10}  {'Diff':>6}")
            for r in model_rows:
                intro_str = f"{r['intro_end']:>9}" if r['intro_end'] is not None else "      n/a"
                print(f"  {str(r['story_id']):>8}  {intro_str}  {r['llm_revelation']:>6}  {r['threshold_revelation']:>10}  {r['diff']:>+6}")
            print(f"  Mean diff (threshold - LLM): {np.mean(diffs):+.2f}  "
                  f"MAE: {np.mean(abs_diffs):.2f}  "
                  f"Std: {np.std(diffs):.2f}")
            intro_rows = [r for r in model_rows if r['intro_end'] is not None]
            if intro_rows:
                n_rev_after_intro = sum(1 for r in intro_rows if r['llm_revelation'] > r['intro_end'])
                print(f"  LLM revelation after intro end: {n_rev_after_intro}/{len(intro_rows)}"
                      f"  (mean intro_end={np.mean([r['intro_end'] for r in intro_rows]):.1f})")
            if len(model_rows) >= 3:
                r_val, p_val = scipy_stats.pearsonr(llm_vals, thr_vals)
                print(f"  Pearson r: {r_val:.3f}  (p={p_val:.3f})")

        # Overall summary
        all_llm = [r['llm_revelation'] for r in rows]
        all_thr = [r['threshold_revelation'] for r in rows]
        all_diffs = [r['diff'] for r in rows]
        all_abs = [r['abs_diff'] for r in rows]
        print(f"\nOVERALL  (n={len(rows)}):")
        print(f"  Mean diff (threshold - LLM): {np.mean(all_diffs):+.2f}")
        print(f"  MAE: {np.mean(all_abs):.2f}  Std: {np.std(all_diffs):.2f}")
        if len(rows) >= 3:
            r_val, p_val = scipy_stats.pearsonr(all_llm, all_thr)
            print(f"  Pearson r: {r_val:.3f}  (p={p_val:.3f})")
        print(f"  Threshold >= LLM: {sum(1 for d in all_diffs if d >= 0)}/{len(rows)}")
        all_intro_rows = [r for r in rows if r['intro_end'] is not None]
        if all_intro_rows:
            n_after = sum(1 for r in all_intro_rows if r['llm_revelation'] > r['intro_end'])
            mean_intro = np.mean([r['intro_end'] for r in all_intro_rows])
            print(f"  LLM revelation after intro end: {n_after}/{len(all_intro_rows)}"
                  f"  (mean intro_end={mean_intro:.1f})")

        return rows

    def get_experienced_reader_fairplay_scores(self, setting=2):
        """Collect per-story fairplay scores for the experienced reader.

        Settings 0-2 use -es2 data; settings 3-5 use -es3 data.

        Returns a dict mapping (model_name, story_type, story_id) to the
        fairplay score (x25), or an empty dict if no data.
        """
        scores = {}
        data_store = self.experienced_reader_data if setting <= 2 else self.experienced_reader_data_es3
        if not data_store:
            return scores

        for key, value in data_store.items():
            model_name, story_type, story_id = key

            exp_dist, true_idx = self.get_experienced_reader_distributions(
                model_name, story_type, story_id, setting=setting
            )
            if exp_dist is None or true_idx is None:
                continue

            llm_key = (model_name, story_type, story_id)
            if llm_key not in self.llm_data:
                continue

            # Use get_naive_reader_probs to respect the current naive_reader_mode;
            # use its true_idx (from LLM data) to ensure consistency with human/KIA FP.
            is_real = story_type in ['poirot', 'sherlock']
            story_name = f"{story_type} {story_id}" if is_real else f"{model_name} {story_id}"
            naive_dist, naive_true_idx = self.get_naive_reader_probs(story_name)
            if naive_dist is None or naive_true_idx is None or len(naive_dist) != len(exp_dist):
                continue

            exp_true = exp_dist[:, naive_true_idx]
            naive_true = naive_dist[:, naive_true_idx]
            fp = np.mean(exp_true - naive_true) * 25
            scores[key] = fp

        return scores

    def get_experienced_reader_coherence_scores(self, setting=2):
        """Collect per-story coherence scores (mean prob on true culprit × 25) for the experienced reader.

        Returns a dict mapping (model_name, story_type, story_id) to the
        coherence score (x25), or an empty dict if no data.
        """
        scores = {}
        data_store = self.experienced_reader_data if setting <= 2 else self.experienced_reader_data_es3
        if not data_store:
            return scores

        for key in data_store:
            model_name, story_type, story_id = key
            exp_dist, _ = self.get_experienced_reader_distributions(
                model_name, story_type, story_id, setting=setting
            )
            if exp_dist is None:
                continue
            if key not in self.llm_data:
                continue
            # Use true_idx from LLM/naive data to stay consistent with FP calculations
            is_real = story_type in ['poirot', 'sherlock']
            story_name = f"{story_type} {story_id}" if is_real else f"{model_name} {story_id}"
            _, naive_true_idx = self.get_naive_reader_probs(story_name)
            if naive_true_idx is None or len(exp_dist) == 0:
                continue
            scores[key] = float(np.mean(exp_dist[:, naive_true_idx])) * 25

        return scores

    def check_experienced_true_idx(self):
        """Check for mismatches between the true culprit index in experienced-reader
        files and the index used by the naive/LLM data for the same story.

        Prints a table of any discrepancies (story, setting, exp_idx, naive_idx).
        """
        mismatches = []

        all_settings = (
            [(s, self.experienced_reader_data)     for s in range(3)] +
            [(s, self.experienced_reader_data_es3) for s in range(3, 6)]
        )

        for setting, data_store in all_settings:
            if not data_store:
                continue
            for key in data_store:
                model_name, story_type, story_id = key

                exp_dist, exp_true_idx = self.get_experienced_reader_distributions(
                    model_name, story_type, story_id, setting=setting
                )
                if exp_dist is None or exp_true_idx is None:
                    continue

                is_real = story_type in ['poirot', 'sherlock']
                story_name = f"{story_type} {story_id}" if is_real else f"{model_name} {story_id}"
                result = self.get_naive_reader_probs(story_name)
                if result is None or not isinstance(result, tuple):
                    continue
                _, naive_true_idx = result
                if naive_true_idx is None:
                    continue

                if exp_true_idx != naive_true_idx:
                    mismatches.append((story_name, setting, exp_true_idx, naive_true_idx))

        if not mismatches:
            print("No true-culprit-index mismatches between experienced-reader and naive data.")
        else:
            print(f"Found {len(mismatches)} experienced-reader / naive true-idx mismatch(es):")
            print(f"  {'Story':<35}  {'Setting':>7}  {'Exp idx':>7}  {'Naive idx':>9}")
            print("  " + "-" * 63)
            for story_name, setting, exp_idx, naive_idx in sorted(mismatches):
                print(f"  {story_name:<35}  {setting:>7}  {exp_idx:>7}  {naive_idx:>9}")