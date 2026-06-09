"""Plotting methods for DetectiveStoryAnalyzer.

These methods are added to the DetectiveStoryAnalyzer class via a mixin pattern.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import probplot

from config import (
    USE_ARGMAX, REAL_STORY_S_MODEL, EXPERIENCED_READER_LABELS,
    EXPERIENCED_READER_SHORT_LABELS, EXPERIENCED_READER_COLORS,
    model_names_all, model_names_with_human, display_names, ignore
)
from utils import clopper_pearson_interval, argmax_distribution


class PlottingMixin:
    """Mixin class containing all plotting/visualization methods.

    These methods assume the base class provides:
        - self.human_data, self.llm_data, self.experienced_reader_data
        - self.get_human_distribution(), self.get_llm_distributions()
        - self.get_naive_reader_probs(), self.get_experienced_reader_distributions()
        - self.get_experienced_reader_fairplay_scores()
        - self._resolve_llm_key()
        - self.naive_reader_mode

    Set self.save_pdf = True to also save every plot as a PDF alongside the PNG.
    """

    save_pdf: bool = False

    def _savefig(self, path, fig=None):
        """Save figure to path (PNG) and optionally as PDF if self.save_pdf is True."""
        f = fig if fig is not None else plt.gcf()
        f.savefig(path, dpi=300, bbox_inches='tight')
        if self.save_pdf:
            pdf_path = path.rsplit('.', 1)[0] + '.pdf'
            f.savefig(pdf_path, dpi=200, bbox_inches='tight')

    def _savefig_prefix(self, fig, prefix):
        """Save figure as <prefix>.png and (if save_pdf) <prefix>.pdf."""
        fig.savefig(f"{prefix}.png", dpi=300, bbox_inches='tight')
        if self.save_pdf:
            fig.savefig(f"{prefix}.pdf", dpi=300, bbox_inches='tight')

    def plot_llm_fairplay_all_stories(self, save_path=None):
        """Plot LLM fairplay for ALL stories and ALL models, preferring -nsamp 20 files"""

        # Place Gemini 3 models first within the Gemini group, then remaining models in list order
        gemini3 = ['gemini-3.1-pro-preview', 'gemini-3-flash-preview']
        model_names = [m for m in gemini3 if m in model_names_all] + \
                      [m for m in model_names_all if m not in gemini3]

        # Organize data by model, preferring nsamp=20
        model_data = {}
        for key, value in self.llm_data.items():
            model_name, story_type, story_id = key
            nsamp = value['params'].get('nsamp', '5')

            # Create unique story identifier
            story_key = f"{model_name}_{story_type}_{story_id}"

            if model_name not in model_data:
                model_data[model_name] = {}

            # Store or update if this has nsamp=20
            if story_key not in model_data[model_name] or nsamp == '20':
                model_data[model_name][story_key] = (key, value, nsamp)

        llm_fairplay = []
        labels = []

        for model in model_names:
            if model not in model_data:
                continue

            llm_scores = []

            for story_key, (key, value, nsamp) in model_data[model].items():
                model_name, story_type, story_id = key

                # Skip real stories — the LLM did not write them
                if story_type in ['poirot', 'sherlock']:
                    continue

                # Get distributions
                sampling_dist, naive_dist, true_idx, sampling_ci, nsamp_data = self.get_llm_distributions(
                    model_name, story_type, story_id
                )

                if naive_dist is None or sampling_dist is None:
                    continue

                if len(sampling_dist) != len(naive_dist):
                    continue

                if true_idx is None:
                    continue
                # Calculate LLM fair play
                naive_true = naive_dist[:, true_idx]
                sampling_true = sampling_dist[:, true_idx]

                llm_fp = np.mean(sampling_true - naive_true)
                llm_scores.append(llm_fp * 25)

            if len(llm_scores) > 0:
                llm_fairplay.append(llm_scores)
                labels.append(display_names.get(model, model))

        # Create plot
        fig, ax = plt.subplots(figsize=(9, 6))

        bp = ax.boxplot(llm_fairplay, labels=labels, patch_artist=True, showfliers=True)

        # give each model type (i.e., gemini, llama, gpt) its own color from: cyan, light brown, light purple]
        # first color (for all gemini) is cyan, second is light brown, third is light purple
        colors = ['lightblue', '#ace1af', '#d2691e', 'lightcoral', 'plum']
        for patch, label in zip(bp['boxes'], labels):
            if 'gemini' in label.lower():
                patch.set_facecolor(colors[0])
                # patch.set_facecolor('#1f77b4')
            elif 'llama' in label.lower():
                patch.set_facecolor(colors[1])
                # patch.set_facecolor('#2ca02c')
            elif 'gpt' in label.lower():
                patch.set_facecolor(colors[2])
                # patch.set_facecolor('#d95f02')
            else:
                patch.set_facecolor('lightgray')

        # # Per-family line colors
        # FAMILY_LINE = {
        #     'gemini15': '#1f77b4',
        #     'gemini25': '#4ea8de',
        #     'gemini3':  '#023e8a',
        #     'llama':    '#2ca02c',
        #     'gpt':      '#d95f02',
        #     'other':    '#7f7f7f',
        # }

        # add a dashed vertical line to separate the model types
        for i in range(len(labels) - 1):
            if (('gemini' in labels[i].lower() and 'gemini' not in labels[i + 1].lower()) or
                    ('llama' in labels[i].lower() and 'llama' not in labels[i + 1].lower())):
                ax.axvline(x=i + 1.5, color='black', linestyle='--', alpha=0.5)

        ax.set_ylabel(f'LLM-based Fair Play Score (×L)', fontsize=12)
        if self.naive_reader_mode != "default":
            ax.set_title(f'LLM-based Fair Play Scores Across All Stories - {self.naive_reader_mode}',
                         fontsize=14, fontweight='bold')
        else:
            ax.set_title(f'LLM-based Fair Play Scores Across All Stories',
                         fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            self._savefig(save_path)

        plt.show()

        # Print statistics
        print("\nLLM Fair Play Statistics:")
        for label, scores in zip(labels, llm_fairplay):
            print(f"{label}: mean={np.mean(scores):.3f}, std={np.std(scores):.3f}, n={len(scores)}")


    def plot_poirot_vs_sherlock(self, save_path=None, naive=False, use_argmax=None, show_sampling=True):
        """Plot average probability curves for Poirot vs Sherlock stories with pooled Clopper-Pearson intervals

        Args:
            save_path: Path to save the plot
            naive: If True, plot naive reader curves instead of human curves
            use_argmax: If True, use argmax (majority vote). If False, use population distribution.
                        If None, use global USE_ARGMAX setting.
            show_sampling: If True (default), overlay LLM sampling curves on the plot.
                           If False, only the gullible/human reader curves are shown.
        """
        # Store curves and metadata for pooling
        poirot_data = {'curves': [], 'num_responses': [], 'nsamp': []}
        sherlock_data = {'curves': [], 'num_responses': [], 'nsamp': []}
        poirot_sampling_data = {'curves': [], 'nsamp': []}
        sherlock_sampling_data = {'curves': [], 'nsamp': []}

        real_names = [f"poirot {i}" for i in range(10)] + [f"sherlock {i}" for i in range(10)]

        for story_name in real_names:
            if story_name in ignore:
                continue
            human_dist, ci, num_responses = self.get_human_distribution(story_name, use_argmax=use_argmax)
            if human_dist is None and not naive:
                continue

            naive_dist, true_idx = self.get_naive_reader_probs(story_name, use_argmax=use_argmax)
            if true_idx is None or naive_dist is None or (not naive and len(naive_dist) != len(human_dist)):
                continue

            # Get sampling distribution for real stories
            if show_sampling:
                model_key, story_id = story_name.split()
                if model_key in ['poirot', 'sherlock']:
                    sampling_dist, _, _, sampling_ci, nsamp = self.get_llm_distributions(
                        REAL_STORY_S_MODEL, model_key, story_id, use_argmax=use_argmax
                    )
                    if sampling_dist is not None and (naive or len(sampling_dist) == len(human_dist)):
                        sampling_true_probs = sampling_dist[:, true_idx]
                        if "poirot" in story_name:
                            poirot_sampling_data['curves'].append(sampling_true_probs)
                            poirot_sampling_data['nsamp'].append(nsamp)
                        elif "sherlock" in story_name:
                            sherlock_sampling_data['curves'].append(sampling_true_probs)
                            sherlock_sampling_data['nsamp'].append(nsamp)

            if naive:
                true_culprit_probs = naive_dist[:, true_idx]
                # For naive mode, we treat it as deterministic (no samples to pool)
                if "poirot" in story_name:
                    poirot_data['curves'].append(true_culprit_probs)
                    poirot_data['num_responses'].append(1)  # Deterministic
                elif "sherlock" in story_name:
                    sherlock_data['curves'].append(true_culprit_probs)
                    sherlock_data['num_responses'].append(1)  # Deterministic
            else:
                true_culprit_probs = human_dist[:, true_idx]
                if "poirot" in story_name:
                    poirot_data['curves'].append(true_culprit_probs)
                    poirot_data['num_responses'].append(num_responses)
                elif "sherlock" in story_name:
                    sherlock_data['curves'].append(true_culprit_probs)
                    sherlock_data['num_responses'].append(num_responses)

        num_points = 25

        def interpolate_curves(curves):
            """Interpolate curves to common number of points"""
            interpolated = []
            for curve in curves:
                x_orig = np.linspace(0, 1, len(curve))
                x_new = np.linspace(0, 1, num_points)
                curve_interp = np.interp(x_new, x_orig, curve)
                interpolated.append(curve_interp)
            return np.array(interpolated)

        def calculate_pooled_ci(data, is_sampling=False):
            """Calculate pooled Clopper-Pearson confidence intervals across stories"""
            if len(data['curves']) == 0:
                return None, None, None

            curves_array = interpolate_curves(data['curves'])
            mean_curve = np.mean(curves_array, axis=0)

            # For each timestep, pool all predictions across stories
            ci_lower = []
            ci_upper = []

            for timestep_idx in range(num_points):
                if is_sampling:
                    # For sampling: sum counts across stories
                    total_count = 0
                    total_trials = 0
                    for story_idx, nsamp in enumerate(data['nsamp']):
                        prob = curves_array[story_idx, timestep_idx]
                        count = int(round(prob * nsamp))
                        total_count += count
                        total_trials += nsamp
                else:
                    # For human/naive: sum counts across stories
                    total_count = 0
                    total_trials = 0
                    for story_idx, n_resp in enumerate(data['num_responses']):
                        prob = curves_array[story_idx, timestep_idx]
                        count = int(round(prob * n_resp))
                        total_count += count
                        total_trials += n_resp

                lower, upper = clopper_pearson_interval(total_count, total_trials)
                ci_lower.append(lower)
                ci_upper.append(upper)

            return mean_curve, np.array(ci_lower), np.array(ci_upper)

        # Calculate means and pooled CIs
        poirot_mean, poirot_ci_lower, poirot_ci_upper = calculate_pooled_ci(poirot_data, is_sampling=False)
        sherlock_mean, sherlock_ci_lower, sherlock_ci_upper = calculate_pooled_ci(sherlock_data, is_sampling=False)
        poirot_sampling_mean, poirot_sampling_ci_lower, poirot_sampling_ci_upper = calculate_pooled_ci(
            poirot_sampling_data, is_sampling=True)
        sherlock_sampling_mean, sherlock_sampling_ci_lower, sherlock_sampling_ci_upper = calculate_pooled_ci(
            sherlock_sampling_data, is_sampling=True)

        # Plotting
        plt.figure(figsize=(10, 6))
        x = np.linspace(0, 1, num_points)

        n_poirot = len(poirot_data['curves'])
        n_sherlock = len(sherlock_data['curves'])

        if poirot_mean is not None:
            plt.plot(x, poirot_mean, 'r-', label=f'Poirot (n={n_poirot})', linewidth=2.2)
            plt.fill_between(x, poirot_ci_lower, poirot_ci_upper,
                             color='red', alpha=0.15, edgecolor='red', linewidth=0)

        if sherlock_mean is not None:
            plt.plot(x, sherlock_mean, 'b--', label=f'Sherlock (n={n_sherlock})', linewidth=2.2,
                     dashes=(6, 3))
            plt.fill_between(x, sherlock_ci_lower, sherlock_ci_upper,
                             facecolor='none', edgecolor='blue', alpha=0.35,
                             hatch='///', linewidth=0)

        if poirot_sampling_mean is not None:
            plt.plot(x, poirot_sampling_mean, 'rx-', label=f'Poirot Sampling (n={len(poirot_sampling_data["curves"])})',
                     linewidth=1.5, markersize=4, markevery=2)
            plt.fill_between(x, poirot_sampling_ci_lower, poirot_sampling_ci_upper,
                             color='red', alpha=0.1, hatch='...', edgecolor='red', linewidth=0)

        if sherlock_sampling_mean is not None:
            plt.plot(x, sherlock_sampling_mean, 'b.:', label=f'Sherlock Sampling (n={len(sherlock_sampling_data["curves"])})',
                     linewidth=1.5, markersize=6, markevery=2)
            plt.fill_between(x, sherlock_sampling_ci_lower, sherlock_sampling_ci_upper,
                             color='blue', alpha=0.1, hatch='...', edgecolor='blue', linewidth=0)

        plt.xlabel('Relative Time in Story', fontsize=16)
        plt.ylabel('Average Probability of True Culprit', fontsize=16)
        apply_argmax = USE_ARGMAX if use_argmax is None else use_argmax
        argmax_label = " (Majority Vote)" if apply_argmax else " (Population Dist.)"
        if self.naive_reader_mode != "default":
            title = f'Poirot vs Sherlock Stories — {"Gullible Reader" if naive else "Actual Reader"}{argmax_label} [{self.naive_reader_mode}]'
        else:
            title = f'Poirot vs Sherlock Stories — {"Gullible Reader" if naive else "Actual Reader"}'
        plt.title(title, fontsize=18, fontweight='bold')
        # legend in the upper left
        plt.legend(fontsize=16, loc='upper left')
        plt.tick_params(axis='x', labelsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        plt.tight_layout()

        if save_path:
            self._savefig(save_path)

        plt.show()

        print(f"Poirot stories analyzed: {len(poirot_data['curves'])}, respondents per story: {poirot_data['num_responses']}")
        print(f"Sherlock stories analyzed: {len(sherlock_data['curves'])}, respondents per story: {sherlock_data['num_responses']}")
        print(f"Poirot sampling curves: {len(poirot_sampling_data['curves'])}")
        print(f"Sherlock sampling curves: {len(sherlock_sampling_data['curves'])}")


    def plot_fairplay_comparison(self, save_path=None, experienced_settings=None, with_human=True):
        """Create whisker plot comparing human-based, LLM-based, and (optionally)
        experienced-reader fair play scores per model.

        Args:
            save_path: Path to save the plot.
            experienced_settings: List of experienced-reader settings to include
                as additional box columns.  Each entry is an int (0–5):
                    From -es2 files:
                        0 = predict culprit (general instructions, 0-shot)
                        1 = 9 previous stories given
                        2 = 3 previous stories given
                    From -es3 files:
                        3 = different prompt, 0-shot alt
                        4 = 2 previous stories given
                        5 = 5 previous stories given
                Pass ``None`` (default) to omit experienced-reader bars.
            with_human: If True (default), plot human fair play scores and restrict
                to stories that have human data.  If False, omit human scores and
                use ALL stories with LLM data for the experienced-reader comparison.
        """
        if experienced_settings is None:
            experienced_settings = []

        model_names = model_names_all

        # Number of box columns per model group
        n_base = (1 if with_human else 0) + 1  # (human + ) know-it-all
        n_exp = len(experienced_settings)
        n_cols = n_base + n_exp
        group_width = n_cols * 0.75 + 0.5

        human_fairplay = []   # only populated when with_human=True
        llm_fairplay = []
        experienced_fairplay = {s: [] for s in experienced_settings}
        labels = []

        # Pre-fetch experienced reader score dicts for stories not requiring human data
        if not with_human and experienced_settings:
            exp_score_dicts = {s: self.get_experienced_reader_fairplay_scores(setting=s)
                               for s in experienced_settings}

        for model in model_names:
            human_scores = []
            llm_scores = []
            exp_scores_per_setting = {s: [] for s in experienced_settings}

            if with_human:
                # ---- WITH HUMAN: iterate over human_data stories for this model ----
                for story_name in self.human_data.keys():
                    if story_name.split()[0] != model:
                        continue

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

                    human_fp = np.mean(human_true - naive_true)
                    human_scores.append(human_fp * 25)

                    llm_fp = np.mean(sampling_true - naive_true)
                    llm_scores.append(llm_fp * 25)

                    # Experienced reader scores for each requested setting
                    # Use true_idx from naive (already in scope) for consistency with human/KIA FP
                    llm_key, _ = self._resolve_llm_key(story_name)
                    for setting in experienced_settings:
                        exp_dist, _ = self.get_experienced_reader_distributions(
                            *llm_key, setting=setting
                        )
                        if exp_dist is not None and len(exp_dist) == len(naive_dist):
                            exp_true = exp_dist[:, true_idx]
                            exp_fp = np.mean(exp_true - naive_true) * 25
                            exp_scores_per_setting[setting].append(exp_fp)

            else:
                # ---- WITHOUT HUMAN: iterate over ALL LLM stories for this model ----
                for key, value in self.llm_data.items():
                    llm_model_name, story_type, story_id = key
                    if llm_model_name != model:
                        continue
                    # Skip real stories — the LLM did not write them
                    if story_type in ['poirot', 'sherlock']:
                        continue

                    sampling_dist, _, true_idx, _, _ = self.get_llm_distributions(
                        llm_model_name, story_type, story_id
                    )
                    if sampling_dist is None or true_idx is None:
                        continue

                    # Use get_naive_reader_probs to respect the current naive_reader_mode
                    story_name = f"{llm_model_name} {story_id}"
                    naive_dist, _ = self.get_naive_reader_probs(story_name)
                    if naive_dist is None or len(naive_dist) != len(sampling_dist):
                        continue

                    naive_true = naive_dist[:, true_idx]
                    sampling_true = sampling_dist[:, true_idx]
                    llm_fp = np.mean(sampling_true - naive_true)
                    llm_scores.append(llm_fp * 25)

                    # Experienced reader
                    for setting in experienced_settings:
                        if key in exp_score_dicts[setting]:
                            exp_scores_per_setting[setting].append(exp_score_dicts[setting][key])

            has_data = len(llm_scores) > 0 if not with_human else len(human_scores) > 0
            if has_data:
                if with_human:
                    human_fairplay.append(human_scores)
                llm_fairplay.append(llm_scores)
                for setting in experienced_settings:
                    experienced_fairplay[setting].append(exp_scores_per_setting[setting])
                labels.append(display_names.get(model, model))

        # ---- Print summary ----
        if with_human:
            print("\nLLM Surprisal (mean naive true probability):")
            for label, l_scores in zip(labels, llm_fairplay):
                print(f"{label}: mean={np.mean(l_scores) / 25:.3f}, n={len(l_scores)}")

            print("\nFair Play Comparison:")
            for i, label in enumerate(labels):
                parts = [f"Human={np.mean(human_fairplay[i]) / 25:.3f}",
                         f"KnowItAll={np.mean(llm_fairplay[i]) / 25:.3f}"]
                for setting in experienced_settings:
                    scores = experienced_fairplay[setting][i]
                    short = EXPERIENCED_READER_SHORT_LABELS[setting]
                    if scores:
                        parts.append(f"Exp({short})={np.mean(scores) / 25:.3f} (n={len(scores)})")
                    else:
                        parts.append(f"Exp({short})=N/A")
                print(f"{label}: {', '.join(parts)}, n_stories={len(human_fairplay[i])}")
        else:
            print("\nFair Play Comparison (no human, all stories):")
            for i, label in enumerate(labels):
                parts = [f"KnowItAll={np.mean(llm_fairplay[i]) / 25:.3f}"]
                for setting in experienced_settings:
                    scores = experienced_fairplay[setting][i]
                    short = EXPERIENCED_READER_SHORT_LABELS[setting]
                    if scores:
                        parts.append(f"Exp({short})={np.mean(scores) / 25:.3f} (n={len(scores)})")
                    else:
                        parts.append(f"Exp({short})=N/A")
                print(f"{label}: {', '.join(parts)}, n_stories={len(llm_fairplay[i])}")

        # ---- Build the plot ----
        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 6))

        positions = []
        all_data = []
        colors = []

        for i in range(len(labels)):
            base_pos = i * group_width
            col = 0

            if with_human:
                positions.append(base_pos + col * 0.75)
                all_data.append(human_fairplay[i])
                colors.append('lightcoral')
                col += 1

            # Know-it-all
            positions.append(base_pos + col * 0.75)
            all_data.append(llm_fairplay[i])
            colors.append('lightblue')
            col += 1

            # Experienced reader(s)
            for setting in experienced_settings:
                scores = experienced_fairplay[setting][i]
                positions.append(base_pos + col * 0.75)
                all_data.append(scores if scores else [])
                colors.append(EXPERIENCED_READER_COLORS.get(setting, 'lightyellow'))
                col += 1

        # Filter out empty boxes (boxplot can't handle empty lists)
        valid_mask = [len(d) > 0 for d in all_data]
        valid_positions = [p for p, v in zip(positions, valid_mask) if v]
        valid_data = [d for d, v in zip(all_data, valid_mask) if v]
        valid_colors = [c for c, v in zip(colors, valid_mask) if v]

        if valid_data:
            bp = ax.boxplot(valid_data, positions=valid_positions, widths=0.6,
                            patch_artist=True, showfliers=True)
            for patch, color in zip(bp['boxes'], valid_colors):
                patch.set_facecolor(color)

        # For empty experienced reader slots, draw a small "no data" marker
        for pos, has_data in zip(positions, valid_mask):
            if not has_data:
                ax.plot(pos, 0, 'x', color='gray', markersize=8, markeredgewidth=1.5)
                ax.annotate('n/a', (pos, 0), textcoords="offset points",
                            xytext=(0, -12), ha='center', fontsize=7, color='gray')

        # X-axis labels centered on each model group
        center_offset = (n_cols - 1) * 0.75 / 2
        label_positions = [i * group_width + center_offset for i in range(len(labels))]
        ax.set_xticks(label_positions)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        # Dashed vertical lines between model families
        for i in range(len(labels) - 1):
            if (('gemini' in labels[i].lower() and 'gemini' not in labels[i + 1].lower()) or
                    ('llama' in labels[i].lower() and 'llama' not in labels[i + 1].lower())):
                mid = (i * group_width + (n_cols - 1) * 0.75 + (i + 1) * group_width) / 2
                ax.axvline(x=mid, color='black', linestyle='--', alpha=0.5)

        # Legend
        legend_elements = []
        if with_human:
            legend_elements.append(Patch(facecolor='lightcoral', label='Human Fair Play'))
        legend_elements.append(Patch(facecolor='lightblue', label='Know-it-all Fair Play'))
        for setting in experienced_settings:
            short = EXPERIENCED_READER_SHORT_LABELS[setting]
            legend_elements.append(
                Patch(facecolor=EXPERIENCED_READER_COLORS.get(setting, 'lightyellow'),
                      label=f'Experienced ({short})')
            )
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

        ax.set_ylabel('Fair Play Score (×L)', fontsize=12)
        title_parts = []
        if with_human:
            title_parts.append('Human vs Know-it-all')
        else:
            title_parts.append('Know-it-all')
        if experienced_settings:
            title_parts.append('vs Experienced Reader')
        scope = '(stories w/ human data)' if with_human else '(all stories)'
        ax.set_title(
            f'Fair Play Comparison: {" ".join(title_parts)} {scope} — {self.naive_reader_mode}',
            fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Threshold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()

        if save_path:
            self._savefig(save_path)

        plt.show()


    # =========================================================================
    # Actual fair play (human vs naive, exp reader vs naive)
    # =========================================================================

    def plot_actual_fairplay(self, exp_setting=0, save_path=None):
        """Bar plot comparing actual (human) fair play and experienced reader fair play,
        both measured against the default naive reader baseline.

        Human FP is computed only for stories that have human response data.
        Experienced reader FP is computed for ALL stories with experienced reader data
        (typically a larger set).  The naive baseline is always 'default' mode.

        Args:
            exp_setting: Which experienced reader setting to use (0=0-shot, 3=0-shot-alt, etc.)
            save_path: Path to save the plot PNG.
        """
        saved_mode = self.naive_reader_mode
        self.set_naive_reader_mode("default")

        # Pre-fetch all experienced reader scores (all stories)
        exp_scores_dict = self.get_experienced_reader_fairplay_scores(setting=exp_setting)

        labels = []
        human_fp_lists = []
        exp_fp_lists = []

        for model in model_names_all:
            human_scores = []
            exp_scores = []

            # --- Human FP: only stories with human data ---
            for story_name in self.human_data.keys():
                if story_name.split()[0] != model:
                    continue
                human_dist, _, _ = self.get_human_distribution(story_name)
                if human_dist is None:
                    continue
                naive_dist, true_idx = self.get_naive_reader_probs(story_name)
                if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                    continue
                human_true = human_dist[:, true_idx]
                naive_true = naive_dist[:, true_idx]
                human_scores.append(np.mean(human_true - naive_true) * 25)

            # --- Exp reader FP: all artificial stories for this model ---
            for (mn, st, sid), fp in exp_scores_dict.items():
                if mn == model and st not in ['poirot', 'sherlock']:
                    exp_scores.append(fp)

            if human_scores or exp_scores:
                human_fp_lists.append(human_scores)
                exp_fp_lists.append(exp_scores)
                labels.append(display_names.get(model, model))

        # ---- Print report ----
        setting_label = EXPERIENCED_READER_SHORT_LABELS.get(exp_setting, str(exp_setting))
        print(f"\nActual Fair Play vs Default Naive (exp setting: {setting_label})")
        print("-" * 70)
        for i, label in enumerate(labels):
            h, e = human_fp_lists[i], exp_fp_lists[i]
            h_str = f"{np.mean(h):.2f} (n={len(h)})" if h else "N/A"
            e_str = f"{np.mean(e):.2f} (n={len(e)})" if e else "N/A"
            print(f"  {label:<30}  Human FP: {h_str:<18}  Exp Reader FP: {e_str}")

        # ---- Plot ----
        n_cols = 2  # human + exp reader
        group_width = n_cols * 0.75 + 0.5

        fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 6))

        positions, all_data, colors = [], [], []
        for i in range(len(labels)):
            base = i * group_width
            positions.append(base)
            all_data.append(human_fp_lists[i])
            colors.append('lightcoral')
            positions.append(base + 0.75)
            all_data.append(exp_fp_lists[i])
            colors.append('#a8d8a8')   # light green

        valid_mask = [len(d) > 0 for d in all_data]
        vpos = [p for p, v in zip(positions, valid_mask) if v]
        vdata = [d for d, v in zip(all_data, valid_mask) if v]
        vcols = [c for c, v in zip(colors, valid_mask) if v]

        # Draw "no data" markers for missing boxes
        for pos, has in zip(positions, valid_mask):
            if not has:
                ax.plot(pos, 0, 'x', color='gray', markersize=8, markeredgewidth=1.5)

        if vdata:
            bp = ax.boxplot(vdata, positions=vpos, widths=0.6,
                            patch_artist=True, showfliers=True)
            for patch, color in zip(bp['boxes'], vcols):
                patch.set_facecolor(color)

        # Annotate each box with n
        for pos, data in zip(positions, all_data):
            if data:
                ax.annotate(f'n={len(data)}', xy=(pos, ax.get_ylim()[0]),
                            xytext=(0, -18), textcoords='offset points',
                            ha='center', fontsize=7, color='gray')

        center_offset = 0.75 / 2
        ax.set_xticks([i * group_width + center_offset for i in range(len(labels))])
        ax.set_xticklabels(labels, rotation=45, ha='right')

        ax.axhline(y=0, color='black', linestyle='--', alpha=0.4)
        ax.set_ylabel('Fair Play Score (×L)', fontsize=12)
        ax.set_title(
            f'Actual Fair Play vs Default Naive Reader\n'
            f'Human (stories w/ responses) vs Experienced Reader [{setting_label}] (all stories)',
            fontsize=13, fontweight='bold'
        )
        legend_elements = [
            Patch(facecolor='lightcoral', label='Human Fair Play'),
            Patch(facecolor='#a8d8a8', label=f'Exp Reader Fair Play ({setting_label})'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        if save_path:
            self._savefig(save_path)
        plt.show()

        self.set_naive_reader_mode(saved_mode)

    # =========================================================================
    # PNAS figures
    # =========================================================================

    def plot_pnas_figures(
        self,
        naive_mode="super_naive_threshold",
        exp_setting=5,
        bump_exp_setting=None,
        save_prefix="pnas",
        fig1_models=None,
        fig1b_models=None,
    ):
        """Generate Figure 1 (two-panel whisker) and Figure 2 (bump/bar chart) for PNAS.

        Args:
            naive_mode: Naive reader mode used as baseline for FP_UB and S.
            exp_setting: Experienced reader setting index for Figure 1 panel B (default 5 = 5-prev).
            bump_exp_setting: Experienced reader setting for the bump chart's Exp FP column.
                              Defaults to exp_setting if not specified.
            save_prefix: Filename prefix; files are saved as
                         <prefix>_fig1.png/pdf and <prefix>_fig2.png/pdf.
            fig1_models: Ordered list of model keys for panel A/Figure 2.
                         Defaults to all models minus Llama-3.2-1B plus Gemini-3.
            fig1b_models: Ordered list of model keys for panel B (must have human data).
                          Defaults to model_names_with_human minus Gemini-3.
        """
        if bump_exp_setting is None:
            bump_exp_setting = exp_setting
        import matplotlib
        matplotlib.rcParams.update({'font.size': 10})

        if fig1_models is None:
            fig1_models = [
                'gemini-1.5-flash', 'gemini-1.5-pro',
                'gemini-2.5-flash-tb0', 'gemini-2.5-flash-tb-1', 'gemini-2.5-pro',
                'gemini-3-flash-preview', 'gemini-3.1-pro-preview',
                'Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct',
                'Llama-3.1-70B-Instruct', 'Llama-3.3-70B-Instruct',
                'gpt-4o-mini', 'gpt-4o',
            ]
        if fig1b_models is None:
            fig1b_models = [m for m in model_names_with_human]

        saved_mode = self.naive_reader_mode
        self.set_naive_reader_mode(naive_mode)

        # ------------------------------------------------------------------
        # Data collection helpers
        # ------------------------------------------------------------------
        def _fp_ub_scores_no_human(model):
            """FP_UB per story for one model, using all its LLM-written stories."""
            scores, s_scores, c_scores = [], [], []
            for key in self.llm_data:
                llm_model, story_type, story_id = key
                if llm_model != model or story_type in ('poirot', 'sherlock'):
                    continue
                sampling_dist, _, true_idx, _, _ = self.get_llm_distributions(
                    llm_model, story_type, story_id)
                if sampling_dist is None or true_idx is None:
                    continue
                story_name = f"{llm_model} {story_id}"
                naive_dist, _ = self.get_naive_reader_probs(story_name)
                if naive_dist is None or len(naive_dist) != len(sampling_dist):
                    continue
                nt = naive_dist[:, true_idx]
                st = sampling_dist[:, true_idx]
                scores.append(np.mean(st - nt) * 25)
                s_scores.append(np.mean(nt) * 25)
                c_scores.append(np.mean(st) * 25)
            return scores, s_scores, c_scores

        def _fp_ub_scores_with_human(model):
            """FP_UB, human FP, and exp FP for stories in human_data for this model."""
            fp_ub, human_fp, exp_fp = [], [], []
            for story_name in self.human_data:
                if story_name.split()[0] != model:
                    continue
                human_dist, _, _ = self.get_human_distribution(story_name)
                if human_dist is None:
                    continue
                naive_dist, true_idx = self.get_naive_reader_probs(story_name)
                if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                    continue
                model_key, story_id = story_name.split()
                model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
                sampling_dist, _, _, _, _ = self.get_llm_distributions(
                    model_key, model_key2, story_id)
                if sampling_dist is None or len(sampling_dist) != len(human_dist):
                    continue
                nt = naive_dist[:, true_idx]
                st = sampling_dist[:, true_idx]
                ht = human_dist[:, true_idx]
                fp_ub.append(np.mean(st - nt) * 25)
                human_fp.append(np.mean(ht - nt) * 25)
                # Experienced reader
                llm_key, _ = self._resolve_llm_key(story_name)
                exp_dist, exp_idx = self.get_experienced_reader_distributions(
                    *llm_key, setting=exp_setting)
                if (exp_dist is not None and exp_idx is not None
                        and len(exp_dist) == len(naive_dist)):
                    exp_fp.append(np.mean(exp_dist[:, exp_idx] - nt) * 25)
            return fp_ub, human_fp, exp_fp

        # ------------------------------------------------------------------
        # Collect panel A data
        # ------------------------------------------------------------------
        panel_a = {}  # model -> (fp_ub_list, s_list, c_list)
        for m in fig1_models:
            fp, s, c = _fp_ub_scores_no_human(m)
            if fp:
                panel_a[m] = (fp, s, c)

        # Sort by mean FP_UB ascending (worst → best)
        sorted_models_a = sorted(
            panel_a, key=lambda m: np.mean(panel_a[m][0]))
        labels_a = [display_names.get(m, m) for m in sorted_models_a]

        # ------------------------------------------------------------------
        # Collect panel B data (same story subset: only stories with human data)
        # ------------------------------------------------------------------
        panel_b = {}  # model -> (fp_ub_list, human_fp_list, exp_fp_list)
        for m in fig1b_models:
            fp, hfp, efp = _fp_ub_scores_with_human(m)
            if fp:
                panel_b[m] = (fp, hfp, efp)

        # Order panel B models by panel A FP_UB mean (only those in panel A)
        sorted_models_b = [m for m in sorted_models_a if m in panel_b]
        # Append any panel-B-only models at the end
        sorted_models_b += [m for m in fig1b_models if m in panel_b and m not in sorted_models_a]
        labels_b = [display_names.get(m, m) for m in sorted_models_b]

        # ------------------------------------------------------------------
        # Figure 1 — two-panel whisker plot
        # ------------------------------------------------------------------
        fig1, (ax_a, ax_b) = plt.subplots(
            1, 2, figsize=(14, 5),
            gridspec_kw={'width_ratios': [4, 5]}
        )

        COLOR_HUMAN = '#f4a8a8'  # light coral — human FP
        COLOR_EXP = '#a8d8a8'  # light green — experienced reader

        # Per-family color and hatch — applied to boxes in Panel A;
        # hatch is reused in Panel B so the model family stays readable.
        FAMILY_COLOR = {
            'gemini15': '#FFD580',  # gold
            'gemini25': '#FF9F40',  # orange
            'gemini3':  '#FF6B6B',  # red
            'llama':    '#7BC8F6',  # sky blue
            'gpt':      '#95E08A',  # green
            'other':    '#C8C8C8',  # grey
        }
        FAMILY_HATCH = {
            'gemini15': '',
            'gemini25': '//',
            'gemini3':  'xx',
            'llama':    '\\\\',
            'gpt':      '--',
            'other':    '.',
        }

        def _family(model_key):
            if 'gemini-1.5' in model_key: return 'gemini15'
            if 'gemini-2.5' in model_key: return 'gemini25'
            if 'gemini-3'   in model_key: return 'gemini3'
            if 'Llama'      in model_key: return 'llama'
            if 'gpt'        in model_key: return 'gpt'
            return 'other'

        def _draw_boxes(ax, models, groups_per_model, metric_colors, labels, title):
            """Draw grouped boxplots.

            models:          list of model keys (for family lookup).
            groups_per_model: list[list[scores]] — one inner list per metric per model.
            metric_colors:   list of colors, one per metric (column within each model group).
                             If None, use the model family color (Panel A behaviour).
            """
            use_family_color = (metric_colors is None)
            n_cols = len(groups_per_model[0]) if groups_per_model else 1
            group_width = n_cols * 0.8 + 0.4
            positions, all_data, all_colors, all_hatches = [], [], [], []
            for i, (m, model_groups) in enumerate(zip(models, groups_per_model)):
                fam = _family(m)
                fam_color = FAMILY_COLOR[fam]
                fam_hatch = FAMILY_HATCH[fam]
                base = i * group_width
                for j, vals in enumerate(model_groups):
                    positions.append(base + j * 0.8)
                    all_data.append(vals)
                    all_colors.append(fam_color if use_family_color else metric_colors[j])
                    all_hatches.append(fam_hatch)
            valid_mask = [len(d) > 0 for d in all_data]
            vpos    = [p for p, v in zip(positions,   valid_mask) if v]
            vdata   = [d for d, v in zip(all_data,    valid_mask) if v]
            vcols   = [c for c, v in zip(all_colors,  valid_mask) if v]
            vhatch  = [h for h, v in zip(all_hatches, valid_mask) if v]
            if vdata:
                bp = ax.boxplot(vdata, positions=vpos, widths=0.65,
                                patch_artist=True, showfliers=True,
                                flierprops=dict(marker='o', markersize=3, alpha=0.5))
                for patch, col, hatch in zip(bp['boxes'], vcols, vhatch):
                    patch.set_facecolor(col)
                    patch.set_hatch(hatch)
                    patch.set_alpha(0.85)
            center_offset = (n_cols - 1) * 0.8 / 2
            ax.set_xticks([i * group_width + center_offset for i in range(len(labels))])
            ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylabel('Fair Play Score (×L)', fontsize=11)
            ax.set_title(title, fontsize=11, fontweight='bold')

        # Panel A — one box per model, colored+hatched by model family
        _draw_boxes(
            ax_a,
            sorted_models_a,
            [[panel_a[m][0]] for m in sorted_models_a],
            None,   # → use family color
            labels_a,
            f'(A) FP_UB per model ({naive_mode})',
        )

        # Family legend for panel A
        seen_families = []
        for m in sorted_models_a:
            f = _family(m)
            if f not in seen_families:
                seen_families.append(f)
        family_label = {
            'gemini15': 'Gemini 1.5', 'gemini25': 'Gemini 2.5',
            'gemini3': 'Gemini 3', 'llama': 'Llama', 'gpt': 'GPT', 'other': 'Other',
        }
        legend_a = [
            Patch(facecolor=FAMILY_COLOR[f], hatch=FAMILY_HATCH[f], label=family_label[f])
            for f in seen_families
        ]
        ax_a.legend(handles=legend_a, fontsize=8, loc='lower right')

        # Panel B — three boxes per model; metric color + family hatch
        _draw_boxes(
            ax_b,
            sorted_models_b,
            [[panel_b[m][0], panel_b[m][1], panel_b[m][2]] for m in sorted_models_b],
            ['#a8c8e8', COLOR_HUMAN, COLOR_EXP],   # metric colors
            labels_b,
            f'(B) FP_UB vs. Human vs. Exp. Reader ({exp_setting}-prev)\n'
            '(Gemini-3 excluded: no human data collected)',
        )

        # Legend for panel B: 3 metrics (left col) + 4 families (right col).
        # With ncol=2 and column-by-column fill, 8 items → 4 per column, so we
        # pad the metric column with one invisible spacer to reach 4 items.
        families_in_b = [f for f in seen_families if f in {_family(m) for m in sorted_models_b}]
        metric_patches = [
            Patch(facecolor='#a8c8e8', label='FP_UB (know-it-all)'),
            Patch(facecolor=COLOR_HUMAN, label='Human FP'),
            Patch(facecolor=COLOR_EXP,
                  label=f'Experienced reader ({EXPERIENCED_READER_SHORT_LABELS[exp_setting]})'),
            Patch(visible=False, label=''),   # invisible spacer
        ]
        family_patches = [
            Patch(facecolor='white', edgecolor='grey', hatch=FAMILY_HATCH[f], label=family_label[f])
            for f in families_in_b
        ]
        ax_b.legend(handles=metric_patches + family_patches,
                    fontsize=8, loc='lower right', ncol=2)

        fig1.suptitle('Fair Play Upper Bound — Generated Stories', fontsize=13, fontweight='bold')
        fig1.tight_layout()
        self._savefig_prefix(fig1, f"{save_prefix}_fig1")
        plt.show()

        # ------------------------------------------------------------------
        # Collect Human FP and Exp Reader FP with DEFAULT naive mode
        # (these two metrics always use "default", regardless of naive_mode)
        # ------------------------------------------------------------------
        self.set_naive_reader_mode("default")
        exp_scores_default = self.get_experienced_reader_fairplay_scores(setting=bump_exp_setting)

        bump_human_fp = {}   # model -> mean human FP (default naive)
        bump_exp_fp   = {}   # model -> mean exp reader FP (default naive)

        for m in sorted_models_a:
            h_scores = []
            for story_name in self.human_data:
                if story_name.split()[0] != m:
                    continue
                human_dist, _, _ = self.get_human_distribution(story_name)
                if human_dist is None:
                    continue
                naive_dist, true_idx = self.get_naive_reader_probs(story_name)
                if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                    continue
                ht = human_dist[:, true_idx]
                nt = naive_dist[:, true_idx]
                h_scores.append(np.mean(ht - nt) * 25)
            if h_scores:
                bump_human_fp[m] = np.mean(h_scores)

            e_scores = [fp for (mn, st, sid), fp in exp_scores_default.items()
                        if mn == m and st not in ('poirot', 'sherlock')]
            if e_scores:
                bump_exp_fp[m] = np.mean(e_scores)

        self.set_naive_reader_mode(naive_mode)  # restore for the rest

        # ------------------------------------------------------------------
        # Figure 2 — bump / ranking chart: S, C, FP_UB, Human FP, Exp FP
        # All three objective metrics (S, C, FP_UB) use naive_mode consistently,
        # same as panel A.  The ids-based waypoint fix ensures super_naive_report
        # gives meaningful variation in S.
        # ------------------------------------------------------------------
        means_s  = {m: np.mean(panel_a[m][1]) for m in sorted_models_a}
        means_c  = {m: np.mean(panel_a[m][2]) for m in sorted_models_a}
        means_fp = {m: np.mean(panel_a[m][0]) for m in sorted_models_a}

        # Models present in all 5 metrics (human + exp reader data)
        bump_models = [m for m in sorted_models_a
                       if m in bump_human_fp and m in bump_exp_fp]

        # Rank helper: rank 1 = best (highest mean)
        def _rank(score_dict):
            ordered = sorted(score_dict, key=score_dict.__getitem__, reverse=True)
            return {m: i + 1 for i, m in enumerate(ordered)}

        # S, C, FP_UB ranked among ALL sorted_models_a — all use naive_mode (super_naive_report)
        # S = surprise = 1 - mean(nt); panel_a stores mean(nt)*25, so negate for ranking
        rank_s  = _rank({m: -means_s[m] for m in sorted_models_a})
        rank_c  = _rank({m: means_c[m]  for m in sorted_models_a})
        rank_fp = _rank({m: means_fp[m] for m in sorted_models_a})
        # Human FP and Exp FP: bump models fill the "gaps" left by non-bump models.
        # Non-bump models hold their FP_UB rank positions as invisible placeholders so
        # the reader can see whether bump models preserved their relative ordering.
        n_all   = len(sorted_models_a)
        n_bump  = len(bump_models)
        non_bump_fp_ranks = {m: rank_fp[m] for m in sorted_models_a if m not in bump_models}
        reserved = set(non_bump_fp_ranks.values())
        free_positions = [r for r in range(1, n_all + 1) if r not in reserved]

        def _rank_expanded(score_dict):
            """Rank bump models in the n_all-wide space, skipping reserved slots."""
            ordered = sorted(score_dict, key=score_dict.__getitem__, reverse=True)
            return {m: free_positions[i] for i, m in enumerate(ordered)}

        rank_hfp = _rank_expanded({m: bump_human_fp[m] for m in bump_models})
        rank_efp = _rank_expanded({m: bump_exp_fp[m]   for m in bump_models})

        bump_exp_short = EXPERIENCED_READER_SHORT_LABELS.get(bump_exp_setting, str(bump_exp_setting))
        naive_short = naive_mode.replace('super_naive_', 'SN-').replace('_', '-')
        # Five metric columns, spread apart so the chart reads wide
        x_cols = [0, 2.5, 5, 7.5, 10]
        metrics = [
            f'S\n({naive_short})',       # 1 - mean(naive_true), uses naive_mode
            'C\n(know-it-all)',          # mean(sampling_true), no naive baseline
            f'FP_UB\n({naive_short})',   # mean(sampling - naive), uses naive_mode
            'Human FP\n(default)',
            f'Exp FP [{bump_exp_short}]\n(default)',
        ]

        # Per-family line colors
        FAMILY_LINE = {
            'gemini15': '#1f77b4',
            'gemini25': '#4ea8de',
            'gemini3':  '#023e8a',
            'llama':    '#2ca02c',
            'gpt':      '#d95f02',
            'other':    '#7f7f7f',
        }

        fig2, ax2 = plt.subplots(figsize=(13, 5))

        # Draw all models for the first 3 columns (S, C, FP_UB)
        for m in sorted_models_a:
            fam   = _family(m)
            color = FAMILY_LINE[fam]
            label = display_names.get(m, m)

            if m in bump_models:
                # Full line across all 5 columns
                ranks = [rank_s[m], rank_c[m], rank_fp[m], rank_hfp[m], rank_efp[m]]
                ax2.plot(x_cols, ranks, color=color, linewidth=1.6,
                         alpha=0.85, zorder=2, solid_capstyle='round')
                for x, r in zip(x_cols, ranks):
                    ax2.scatter(x, r, color=color, s=40, zorder=3, alpha=0.85)
                # Labels on both sides
                ax2.text(x_cols[0] - 0.2,  rank_s[m],   label, ha='right', va='center',
                         fontsize=7.5, color=color)
                ax2.text(x_cols[-1] + 0.2, rank_efp[m], label, ha='left',  va='center',
                         fontsize=7.5, color=color)
            else:
                # Only first 3 columns
                partial_x = x_cols[:3]
                partial_r = [rank_s[m], rank_c[m], rank_fp[m]]
                ax2.plot(partial_x, partial_r, color=color, linewidth=1.6,
                         alpha=0.5, zorder=2, solid_capstyle='round', linestyle='--')
                for x, r in zip(partial_x, partial_r):
                    ax2.scatter(x, r, color=color, s=40, zorder=3, alpha=0.5)
                # Label on the left only
                ax2.text(x_cols[0] - 0.2, rank_s[m], label, ha='right', va='center',
                         fontsize=7.5, color=color, alpha=0.6)

        # Dashed vertical separator between super-naive and default-naive columns
        sep_x = (x_cols[2] + x_cols[3]) / 2
        ax2.axvline(sep_x, color='grey', linewidth=1.2, linestyle='--', alpha=0.6, zorder=1)
        ax2.text(sep_x, n_all + 0.4,
                 f'← S, C, FP_UB: all {n_all} models  |  Human FP, Exp FP: {len(bump_models)} models (others hold FP_UB rank as placeholders) →',
                 ha='center', va='top', fontsize=6.5, color='grey', style='italic')

        ax2.set_xticks(x_cols)
        ax2.set_xticklabels(metrics, fontsize=9, fontweight='bold')
        ax2.set_xlim(x_cols[0] - 2.8, x_cols[-1] + 2.8)
        ax2.set_ylim(n_all + 0.5, 0.5)   # rank 1 at top; y-axis spans all models
        ax2.set_yticks(range(1, n_all + 1))
        ax2.set_yticklabels([str(i) for i in range(1, n_all + 1)], fontsize=8)
        ax2.set_ylabel('Rank  (1 = best)', fontsize=10)
        ax2.yaxis.set_ticks_position('none')
        ax2.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax2.grid(axis='y', alpha=0.2, linestyle=':')
        for x in x_cols:
            ax2.axvline(x, color='grey', linewidth=0.8, alpha=0.4, zorder=1)

        ax2.set_title(
            f'Rankings by metric — {n_all} models total, {len(bump_models)} with human & experienced reader data\n'
            f'S & FP_UB: {naive_short} naive  |  C: know-it-all  |  Human & Exp FP: default naive  (all columns ranked on 1–{n_all} scale)',
            fontsize=10, fontweight='bold')

        fig2.tight_layout()
        self._savefig_prefix(fig2, f"{save_prefix}_fig2")
        plt.show()

        self.set_naive_reader_mode(saved_mode)

        print(f"Saved: {save_prefix}_fig1.png/pdf, {save_prefix}_fig2.png/pdf")


    def plot_bump_sc_fp(
        self,
        naive_mode="default",
        save_prefix="pnas",
        fig_models=None,
        fig_b_models=None,
        include_human_fp1=True,
    ):
        """Bump/ranking chart: S mean, C mean, % FP_UB>1 [, % Human FP>1].

        First three columns use all models; Human FP column (optional) uses models with human data.

        Args:
            naive_mode: Baseline for S, C, and FP_UB. Human FP always uses default.
            save_prefix: Filename prefix; file saved as <prefix>_fig2b.png/pdf.
            fig_models: Models for the first three columns.
            fig_b_models: Models for Human FP column (must have human data).
            include_human_fp1: If True (default), add % Human FP>1 as the 4th column.
        """
        from scipy.stats import rankdata as _rankdata

        if fig_models is None:
            fig_models = [
                'gemini-1.5-flash', 'gemini-1.5-pro',
                'gemini-2.5-flash-tb0', 'gemini-2.5-flash-tb-1', 'gemini-2.5-pro',
                'gemini-3-flash-preview', 'gemini-3.1-pro-preview',
                'Llama-3.1-8B-Instruct', 'Llama-3.2-3B-Instruct',
                'Llama-3.1-70B-Instruct', 'Llama-3.3-70B-Instruct',
                'gpt-4o-mini', 'gpt-4o',
            ]
        if fig_b_models is None:
            fig_b_models = list(model_names_with_human)

        saved_mode = self.naive_reader_mode
        self.set_naive_reader_mode(naive_mode)

        # S mean, C mean, % FP_UB>1 per model (all stories, naive_mode)
        mean_c = {}
        mean_s = {}
        pct_fp_ub = {}
        for m in fig_models:
            c_vals, s_vals, count, total = [], [], 0, 0
            for key in self.llm_data:
                llm_model, story_type, story_id = key
                if llm_model != m or story_type in ('poirot', 'sherlock'):
                    continue
                sampling_dist, _, true_idx, _, _ = self.get_llm_distributions(
                    llm_model, story_type, story_id)
                if sampling_dist is None or true_idx is None:
                    continue
                story_name = f"{llm_model} {story_id}"
                naive_dist, _ = self.get_naive_reader_probs(story_name)
                if naive_dist is None or len(naive_dist) != len(sampling_dist):
                    continue
                st = sampling_dist[:, true_idx]
                nt = naive_dist[:, true_idx]
                c_vals.append(float(np.mean(st)) * 25)
                s_vals.append(float(np.mean(nt)) * 25)
                count += int(np.mean(st - nt) * 25 > 1)
                total += 1
            if total:
                mean_c[m]    = np.mean(c_vals)
                mean_s[m]    = np.mean(s_vals)
                pct_fp_ub[m] = count / total * 100

        # % Human FP>1 per model (stories with human data, default naive)
        pct_hfp = {}
        if include_human_fp1:
            self.set_naive_reader_mode("default")
            for m in fig_b_models:
                count, total = 0, 0
                for story_name in self.human_data:
                    if story_name.split()[0] != m:
                        continue
                    human_dist, _, _ = self.get_human_distribution(story_name)
                    if human_dist is None:
                        continue
                    naive_dist, true_idx = self.get_naive_reader_probs(story_name)
                    if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                        continue
                    count += int(np.mean(human_dist[:, true_idx] - naive_dist[:, true_idx]) * 25 > 1)
                    total += 1
                if total:
                    pct_hfp[m] = count / total * 100

        self.set_naive_reader_mode(saved_mode)

        sorted_models = sorted(pct_fp_ub, key=pct_fp_ub.__getitem__)
        n_all = len(sorted_models)
        bump_models = [m for m in sorted_models if m in pct_hfp]
        n_bump = len(bump_models)

        def _rank_avg(score_dict, models):
            scores = np.array([score_dict[m] for m in models])
            return {m: r for m, r in zip(models, _rankdata(-scores, method='average'))}

        def _scale(r, n_src, n_dst):
            if n_src <= 1:
                return r
            return (r - 1) / (n_src - 1) * (n_dst - 1) + 1

        def _spread(label_pos_list, spacing=0.38):
            result = {}
            sorted_list = sorted(label_pos_list, key=lambda t: t[0])
            i = 0
            while i < len(sorted_list):
                j = i
                while j < len(sorted_list) and abs(sorted_list[j][0] - sorted_list[i][0]) < 0.01:
                    j += 1
                group = sorted_list[i:j]
                n = len(group)
                offsets = np.linspace(-(n - 1) / 2 * spacing, (n - 1) / 2 * spacing, n)
                for (y, m), off in zip(group, offsets):
                    result[m] = y + off
                i = j
            return result

        rank_s   = _rank_avg({m: -mean_s[m] for m in sorted_models}, sorted_models)
        rank_c   = _rank_avg(mean_c,    sorted_models)
        rank_fp  = _rank_avg(pct_fp_ub, sorted_models)
        rank_hfp = _rank_avg(pct_hfp,  bump_models) if include_human_fp1 else None

        def rs(rd, m): return _scale(rd[m], n_bump, n_all)

        left_adj  = _spread([(rank_s[m], m) for m in sorted_models])
        if include_human_fp1 and bump_models:
            right_adj = _spread([(rs(rank_hfp, m), m) for m in bump_models])
        else:
            right_adj = _spread([(rank_fp[m], m) for m in sorted_models])

        FAMILY_LINE = {
            'gemini15': '#1f77b4', 'gemini25': '#4ea8de', 'gemini3': '#023e8a',
            'llama': '#2ca02c', 'gpt': '#d95f02', 'other': '#7f7f7f',
        }

        def _family(mk):
            if 'gemini-1.5' in mk: return 'gemini15'
            if 'gemini-2.5' in mk: return 'gemini25'
            if 'gemini-3'   in mk: return 'gemini3'
            if 'Llama'      in mk: return 'llama'
            if 'gpt'        in mk: return 'gpt'
            return 'other'

        def _wrap(name):
            """Split 'Gemini X.X Flash ...' onto two lines to avoid rank overlap."""
            if 'Flash' in name:
                return name.replace(' Flash', '\nFlash', 1)
            return name

        naive_short = naive_mode.replace('super_naive_', 'SN-').replace('_', '-')
        naive_tag  = f'\n({naive_short})' if naive_mode != 'default' else ''
        human_tag  = '\n(default)'        if naive_mode != 'default' else ''
        if include_human_fp1:
            x_cols  = [0, 2.5, 5, 7.5]
            metrics = [
                f'S mean{naive_tag}',
                f'C mean{naive_tag}',
                f'% FP_UB>1{naive_tag}',
                f'% Human FP>1{human_tag}',
            ]
            figwidth = 8
        else:
            x_cols  = [0, 2, 4]
            metrics = [
                f'S mean{naive_tag}',
                f'C mean{naive_tag}',
                f'% FP_UB>1{naive_tag}',
            ]
            figwidth = 6

        fig, ax = plt.subplots(figsize=(figwidth, 8))

        for m in sorted_models:
            color = FAMILY_LINE[_family(m)]
            label = _wrap(display_names.get(m, m))
            if include_human_fp1 and m in bump_models:
                ranks = [rank_s[m], rank_c[m], rank_fp[m], rs(rank_hfp, m)]
                ax.plot(x_cols, ranks, color=color, linewidth=1.6, alpha=0.85,
                        zorder=2, solid_capstyle='round')
                for x, r in zip(x_cols, ranks):
                    ax.scatter(x, r, color=color, s=40, zorder=3, alpha=0.85)
                ax.text(x_cols[0] - 0.15, left_adj[m],  label, ha='right', va='center',
                        fontsize=9, color=color)
                ax.text(x_cols[-1] + 0.15, right_adj[m], label, ha='left',  va='center',
                        fontsize=9, color=color)
            elif include_human_fp1:
                # non-bump model: dashed line for first 3 cols only
                ranks3 = [rank_s[m], rank_c[m], rank_fp[m]]
                ax.plot(x_cols[:3], ranks3, color=color, linewidth=1.6, alpha=0.45,
                        zorder=2, solid_capstyle='round', linestyle='--')
                for x, r in zip(x_cols[:3], ranks3):
                    ax.scatter(x, r, color=color, s=40, zorder=3, alpha=0.45)
                ax.text(x_cols[0] - 0.15, left_adj[m], label, ha='right', va='center',
                        fontsize=9, color=color, alpha=0.45)
            else:
                ranks = [rank_s[m], rank_c[m], rank_fp[m]]
                ax.plot(x_cols, ranks, color=color, linewidth=1.6, alpha=0.85,
                        zorder=2, solid_capstyle='round')
                for x, r in zip(x_cols, ranks):
                    ax.scatter(x, r, color=color, s=40, zorder=3, alpha=0.85)
                ax.text(x_cols[0] - 0.15, left_adj[m],  label, ha='right', va='center',
                        fontsize=9, color=color)
                ax.text(x_cols[-1] + 0.15, right_adj[m], label, ha='left',  va='center',
                        fontsize=9, color=color)

        # if include_human_fp1:
        #     sep_x = (x_cols[2] + x_cols[3]) / 2
        #     ax.axvline(sep_x, color='grey', linewidth=1.2, linestyle='--', alpha=0.6, zorder=1)
        #     ax.text(sep_x, n_all + 0.6,
        #             f'← all {n_all} models  |  {n_bump} with human data →',
        #             ha='center', va='top', fontsize=9, color='grey', style='italic')

        ax.set_xticks(x_cols)
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_xlim(x_cols[0] - 2.2, x_cols[-1] + 2.2)
        ax.set_ylim(n_all + 0.8, 0.2)
        ax.set_yticks(range(1, n_all + 1))
        ax.set_yticklabels([str(i) for i in range(1, n_all + 1)], fontsize=10)
        ax.set_ylabel('Rank  (1 = best)', fontsize=12)
        ax.yaxis.set_ticks_position('none')
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.grid(axis='y', alpha=0.2, linestyle=':')
        for x in x_cols:
            ax.axvline(x, color='grey', linewidth=0.8, alpha=0.4, zorder=1)

        seen_fams = []
        for m in sorted_models:
            f = _family(m)
            if f not in seen_fams:
                seen_fams.append(f)
        fam_label = {
            'gemini15': 'Gemini 1.5', 'gemini25': 'Gemini 2.5', 'gemini3': 'Gemini 3',
            'llama': 'Llama', 'gpt': 'GPT', 'other': 'Other',
        }
        from matplotlib.patches import Patch as _Patch
        ax.legend(
            handles=[_Patch(facecolor=FAMILY_LINE[f], label=fam_label[f]) for f in seen_fams],
            fontsize=10, loc='lower center', ncol=len(seen_fams),
            bbox_to_anchor=(0.5, -0.02))

        naive_desc = f'{naive_short} naive' if naive_mode != 'default' else 'naive'
        if include_human_fp1:
            title = (
                f'Rankings: S/C mean, % FP_UB>1, % Human FP>1 — {n_all} models ({n_bump} with human data)\n'
                f'S, C, FP_UB: {naive_desc}  |  Human FP: {naive_desc}  (S rank 1 = most surprising; ties → avg rank)'
                if naive_mode == 'default' else
                f'Rankings: S/C mean, % FP_UB>1, % Human FP>1 — {n_all} models ({n_bump} with human data)\n'
                f'S, C, FP_UB: {naive_desc}  |  Human FP: default naive  (S rank 1 = most surprising; ties → avg rank)'
            )
        else:
            title = (
                f'Rankings: S/C mean, % FP_UB>1 — {n_all} models\n'
                f'S, C, FP_UB: {naive_desc}  (S rank 1 = most surprising; ties → avg rank)'
            )
        # ax.set_title(title, fontsize=12, fontweight='bold')

        fig.tight_layout()
        self._savefig_prefix(fig, f"{save_prefix}_fig2b")
        plt.show()
        print(f"Saved: {save_prefix}_fig2b.png" + ("/pdf" if self.save_pdf else ""))


    def plot_bump_fp_thresholds(
        self,
        naive_mode="default",
        save_prefix="pnas",
        fig_b_models=None,
        include_human_fp3=False,
    ):
        """Bump/ranking chart: % FP_UB>1, % FP_UB>2, % Human FP>1, % Human FP>2 [, % Human FP>3].

        All columns use only stories that have human data, so the story set is
        identical across all columns. Only models with human data appear.

        Args:
            naive_mode: Baseline for FP_UB. Human FP always uses default.
            save_prefix: Filename prefix; file saved as <prefix>_fig2c.png/pdf.
            fig_b_models: Models to include (must have human data).
            include_human_fp3: If True, add a 5th column for % Human FP>3.
        """
        from scipy.stats import rankdata as _rankdata

        if fig_b_models is None:
            fig_b_models = list(model_names_with_human)

        saved_mode = self.naive_reader_mode

        # Collect metrics per model, iterating only human-data stories
        pct_fp1 = {}
        pct_fp2 = {}
        pct_h1  = {}
        pct_h2  = {}
        pct_h3  = {}
        for m in fig_b_models:
            fp1, fp2, h1, h2, h3, total = 0, 0, 0, 0, 0, 0
            for story_name in self.human_data:
                if story_name.split()[0] != m:
                    continue
                human_dist, _, _ = self.get_human_distribution(story_name)
                if human_dist is None:
                    continue

                # FP_UB uses naive_mode
                self.set_naive_reader_mode(naive_mode)
                model_key, story_id = story_name.split()
                sampling_dist, _, true_idx, _, _ = self.get_llm_distributions(
                    model_key, model_key, story_id)
                naive_dist_fp, _ = self.get_naive_reader_probs(story_name)

                # Human FP uses default naive
                self.set_naive_reader_mode("default")
                naive_dist_h, true_idx_h = self.get_naive_reader_probs(story_name)

                if (sampling_dist is None or true_idx is None
                        or naive_dist_fp is None or len(naive_dist_fp) != len(sampling_dist)
                        or naive_dist_h is None or true_idx_h is None
                        or len(naive_dist_h) != len(human_dist)):
                    continue

                fp_val = np.mean(sampling_dist[:, true_idx]   - naive_dist_fp[:, true_idx])  * 25
                h_val  = np.mean(human_dist[:, true_idx_h]    - naive_dist_h[:, true_idx_h]) * 25
                fp1 += int(fp_val > 1)
                fp2 += int(fp_val > 2)
                h1  += int(h_val  > 1)
                h2  += int(h_val  > 2)
                h3  += int(h_val  > 3)
                total += 1

            if total:
                pct_fp1[m] = fp1 / total * 100
                pct_fp2[m] = fp2 / total * 100
                pct_h1[m]  = h1  / total * 100
                pct_h2[m]  = h2  / total * 100
                pct_h3[m]  = h3  / total * 100

        self.set_naive_reader_mode(saved_mode)

        sorted_models = sorted(pct_fp1, key=pct_fp1.__getitem__)
        n_all = len(sorted_models)

        def _rank_avg(score_dict, models):
            scores = np.array([score_dict[m] for m in models])
            return {m: r for m, r in zip(models, _rankdata(-scores, method='average'))}

        def _spread(label_pos_list, spacing=0.38):
            result = {}
            sorted_list = sorted(label_pos_list, key=lambda t: t[0])
            i = 0
            while i < len(sorted_list):
                j = i
                while j < len(sorted_list) and abs(sorted_list[j][0] - sorted_list[i][0]) < 0.01:
                    j += 1
                group = sorted_list[i:j]
                n = len(group)
                offsets = np.linspace(-(n - 1) / 2 * spacing, (n - 1) / 2 * spacing, n)
                for (y, m), off in zip(group, offsets):
                    result[m] = y + off
                i = j
            return result

        rank_fp1 = _rank_avg(pct_fp1, sorted_models)
        rank_fp2 = _rank_avg(pct_fp2, sorted_models)
        rank_h1  = _rank_avg(pct_h1,  sorted_models)
        rank_h2  = _rank_avg(pct_h2,  sorted_models)
        rank_h3  = _rank_avg(pct_h3,  sorted_models) if include_human_fp3 else None

        last_rank = rank_h3 if include_human_fp3 else rank_h2
        left_adj  = _spread([(rank_fp1[m], m) for m in sorted_models])
        right_adj = _spread([(last_rank[m], m) for m in sorted_models])

        FAMILY_LINE = {
            'gemini15': '#1f77b4', 'gemini25': '#4ea8de', 'gemini3': '#023e8a',
            'llama': '#2ca02c', 'gpt': '#d95f02', 'other': '#7f7f7f',
        }

        def _family(mk):
            if 'gemini-1.5' in mk: return 'gemini15'
            if 'gemini-2.5' in mk: return 'gemini25'
            if 'gemini-3'   in mk: return 'gemini3'
            if 'Llama'      in mk: return 'llama'
            if 'gpt'        in mk: return 'gpt'
            return 'other'

        def _wrap(name):
            """Split 'Gemini X.X Flash ...' onto two lines to avoid rank overlap."""
            if 'Flash' in name:
                return name.replace(' Flash', '\nFlash', 1)
            return name

        naive_short = naive_mode.replace('super_naive_', 'SN-').replace('_', '-')
        naive_tag  = f'\n({naive_short})' if naive_mode != 'default' else ''
        human_tag  = '\n(default)'        if naive_mode != 'default' else ''
        if include_human_fp3:
            x_cols  = [0, 2, 5, 7, 9]
            metrics = [
                f'% FP_UB>1{naive_tag}',
                f'% FP_UB>2{naive_tag}',
                f'% Human FP>1{human_tag}',
                f'% Human FP>2{human_tag}',
                f'% Human FP>3{human_tag}',
            ]
            figwidth = 7
        else:
            x_cols  = [0, 2, 5, 7]
            metrics = [
                f'% FP_UB>1{naive_tag}',
                f'% FP_UB>2{naive_tag}',
                f'% Human FP>1{human_tag}',
                f'% Human FP>2{human_tag}',
            ]
            figwidth = 6

        fig, ax = plt.subplots(figsize=(figwidth, 8))

        for m in sorted_models:
            color = FAMILY_LINE[_family(m)]
            label = _wrap(display_names.get(m, m))
            ranks = [rank_fp1[m], rank_fp2[m], rank_h1[m], rank_h2[m]]
            if include_human_fp3:
                ranks.append(rank_h3[m])
            ax.plot(x_cols, ranks, color=color, linewidth=1.6, alpha=0.85,
                    zorder=2, solid_capstyle='round')
            for x, r in zip(x_cols, ranks):
                ax.scatter(x, r, color=color, s=40, zorder=3, alpha=0.85)
            ax.text(x_cols[0] - 0.15, left_adj[m],  label, ha='right', va='center',
                    fontsize=9, color=color)
            ax.text(x_cols[-1] + 0.15, right_adj[m], label, ha='left',  va='center',
                    fontsize=9, color=color)

        sep_x = (x_cols[1] + x_cols[2]) / 2
        ax.axvline(sep_x, color='grey', linewidth=1.2, linestyle='--', alpha=0.6, zorder=1)

        ax.set_xticks(x_cols)
        ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
        ax.set_xlim(x_cols[0] - 2.2, x_cols[-1] + 2.2)
        ax.set_ylim(n_all + 0.8, 0.2)
        ax.set_yticks(range(1, n_all + 1))
        ax.set_yticklabels([str(i) for i in range(1, n_all + 1)], fontsize=10)
        ax.set_ylabel('Rank  (1 = best = highest %)', fontsize=12)
        ax.yaxis.set_ticks_position('none')
        ax.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
        ax.grid(axis='y', alpha=0.2, linestyle=':')
        for x in x_cols:
            ax.axvline(x, color='grey', linewidth=0.8, alpha=0.4, zorder=1)

        seen_fams = []
        for m in sorted_models:
            f = _family(m)
            if f not in seen_fams:
                seen_fams.append(f)
        fam_label = {
            'gemini15': 'Gemini 1.5', 'gemini25': 'Gemini 2.5', 'gemini3': 'Gemini 3',
            'llama': 'Llama', 'gpt': 'GPT', 'other': 'Other',
        }
        from matplotlib.patches import Patch as _Patch
        ax.legend(
            handles=[_Patch(facecolor=FAMILY_LINE[f], label=fam_label[f]) for f in seen_fams],
            fontsize=10, loc='lower center', ncol=len(seen_fams),
            bbox_to_anchor=(0.5, -0.02))

        naive_desc = f'{naive_short} naive' if naive_mode != 'default' else 'naive'
        human_desc = 'default naive' if naive_mode != 'default' else naive_desc
        ax.set_title(
            f'Rankings: % FP>1/2 and % Human FP>1/{2 if not include_human_fp3 else "2/3"} — '
            f'{n_all} models (stories with human data only)\n'
            f'FP_UB: {naive_desc}  |  Human FP: {human_desc}  (ties → average rank)',
            fontsize=12, fontweight='bold')

        fig.tight_layout()
        self._savefig_prefix(fig, f"{save_prefix}_fig2c")
        plt.show()
        print(f"Saved: {save_prefix}_fig2c.png" + ("/pdf" if self.save_pdf else ""))


    def visualize_mixed_effects_results(self, results, df_individual):
        """
        Create visualizations for mixed effects results.
        """

        if 'surprise' not in results['models'] or not results['models']['surprise'].get('converged'):
            print("No surprise model to visualize")
            return

        surprise_data = df_individual[['participant_id', 'surprise', 'surprise_metric']].dropna()
        model_result = results['models']['surprise']['fitted_model']

        # Extract parameters
        fixed_intercept = model_result.params['Intercept']
        fixed_slope = model_result.params['surprise_metric']
        random_effects = model_result.random_effects

        # Create figure with subplots
        fig = plt.figure(figsize=(18, 6))
        gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.3)

        # =========================================================================
        # Panel 1: Population-level effect (fixed effect)
        # =========================================================================

        ax1 = fig.add_subplot(gs[0, 0])

        # Plot all data points
        ax1.scatter(surprise_data['surprise_metric'],
                    surprise_data['surprise'],
                    alpha=0.3, s=30, color='steelblue', edgecolors='none')

        # Plot fixed effect line
        x_range = np.linspace(surprise_data['surprise_metric'].min(),
                              surprise_data['surprise_metric'].max(), 100)
        y_fixed = fixed_intercept + fixed_slope * x_range
        ax1.plot(x_range, y_fixed, 'r-', linewidth=3,
                 label=f'Population Effect\n(β = {fixed_slope:.2f})')

        # Add confidence interval
        # Approximate 95% CI for predictions
        se_slope = results['models']['surprise']['se']
        y_upper = fixed_intercept + (fixed_slope + 1.96 * se_slope) * x_range
        y_lower = fixed_intercept + (fixed_slope - 1.96 * se_slope) * x_range
        ax1.fill_between(x_range, y_lower, y_upper, alpha=0.2, color='red')

        ax1.set_xlabel('Surprise Metric', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Surprise Rating', fontsize=12, fontweight='bold')
        ax1.set_title('(A) Population-Level Fixed Effect', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10, loc='upper left')
        ax1.grid(alpha=0.3)
        ax1.set_xlim(surprise_data['surprise_metric'].min() - 0.05,
                     surprise_data['surprise_metric'].max() + 0.05)

        # =========================================================================
        # Panel 2: Participant-specific effects (random effects)
        # =========================================================================

        ax2 = fig.add_subplot(gs[0, 1])

        # Select subset of participants to show
        participants = surprise_data['participant_id'].unique()
        n_show = min(15, len(participants))
        np.random.seed(42)
        show_participants = np.random.choice(participants, n_show, replace=False)

        colors = plt.cm.tab20(np.linspace(0, 1, n_show))

        for i, participant in enumerate(show_participants):
            p_data = surprise_data[surprise_data['participant_id'] == participant]

            # Get random intercept
            random_intercept = random_effects.get(participant, [0])[0]

            # Participant-specific line
            y_participant = (fixed_intercept + random_intercept) + fixed_slope * x_range
            ax2.plot(x_range, y_participant, '-', color=colors[i],
                     alpha=0.5, linewidth=1.5)

            # Participant's data points
            ax2.scatter(p_data['surprise_metric'], p_data['surprise'],
                        color=colors[i], s=60, alpha=0.7,
                        edgecolors='black', linewidth=0.5)

        # Population average
        ax2.plot(x_range, y_fixed, 'r--', linewidth=3,
                 label='Population Average', alpha=0.8, zorder=100)

        ax2.set_xlabel('Surprise Metric', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Surprise Rating', fontsize=12, fontweight='bold')
        ax2.set_title(f'(B) Participant-Specific Effects (n={n_show})',
                      fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(surprise_data['surprise_metric'].min() - 0.05,
                     surprise_data['surprise_metric'].max() + 0.05)

        # =========================================================================
        # Panel 3: Distribution of random intercepts
        # =========================================================================

        ax3 = fig.add_subplot(gs[0, 2])

        # Extract all random intercepts
        random_intercepts = [random_effects.get(p, [0])[0] for p in participants]

        # Plot histogram
        ax3.hist(random_intercepts, bins=15, color='steelblue',
                 alpha=0.7, edgecolor='black')
        ax3.axvline(0, color='red', linestyle='--', linewidth=2,
                    label='Population Mean')

        ax3.set_xlabel('Random Intercept Value', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Participants', fontsize=12, fontweight='bold')
        ax3.set_title('(C) Distribution of Participant Baselines',
                      fontsize=13, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3, axis='y')

        # Add text with variance
        variance = np.var(random_intercepts)
        ax3.text(0.95, 0.95, f'Variance = {variance:.2f}\nSD = {np.sqrt(variance):.2f}',
                 transform=ax3.transAxes, fontsize=10,
                 verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self._savefig('mixed_effects_visualization.png')
        print("\n✓ Saved: mixed_effects_visualization.png")
        plt.show()

        # =========================================================================
        # Additional plot: Residuals diagnostic
        # =========================================================================

        # Get fitted values and residuals
        fitted = model_result.fittedvalues
        residuals = model_result.resid

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Residuals vs fitted
        ax1.scatter(fitted, residuals, alpha=0.5, s=30)
        ax1.axhline(0, color='red', linestyle='--', linewidth=2)
        ax1.set_xlabel('Fitted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residuals vs Fitted', fontsize=13, fontweight='bold')
        ax1.grid(alpha=0.3)

        # Q-Q plot for normality
        from scipy.stats import probplot
        probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Normal Q-Q Plot', fontsize=13, fontweight='bold')
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        self._savefig('mixed_effects_diagnostics.png')
        print("✓ Saved: mixed_effects_diagnostics.png")
        plt.show()

        print("\nDiagnostic plots interpretation:")
        print("- Left: Residuals should be randomly scattered around 0")
        print("- Right: Points should follow the diagonal line for normality")


    def plot_reading_curves(self, story_name, save_path=None):
        """Plot the three reading curves for a story using probabilities (not argmax)."""
        model_name, story_id = story_name.split()
        is_artificial = model_name not in ['sherlock', 'poirot']
        model_display = story_name

        # Get human distribution WITHOUT argmax (may be None if no human data loaded)
        human_dist, human_ci, num_responses = self.get_human_distribution(story_name, use_argmax=False)

        # Get LLM distributions WITHOUT argmax
        if not is_artificial:
            llm_key = (REAL_STORY_S_MODEL, model_name, story_id)
            sampling_dist, naive_dist, true_idx, sampling_ci, nsamp = self.get_llm_distributions(
                REAL_STORY_S_MODEL, model_name, story_id, use_argmax=False
            )
        else:
            story_type = model_name.split("-tb")[0]
            llm_key = (model_name, story_type, story_id)
            sampling_dist, naive_dist, true_idx, sampling_ci, nsamp = self.get_llm_distributions(
                model_name, story_type, story_id, use_argmax=False
            )

        if true_idx is None:
            return

        # Get paragraph IDs from llm_data
        paragraph_ids = None
        if llm_key in self.llm_data:
            paragraph_ids = self.llm_data[llm_key]['data'].get('ids', None)

        # Determine n_steps from whatever data is available
        if human_dist is not None:
            n_steps = len(human_dist)
        elif naive_dist is not None:
            n_steps = len(naive_dist)
        elif sampling_dist is not None:
            n_steps = len(sampling_dist)
        else:
            print(f"No data available for {story_name}")
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        # Use paragraph numbers for x-axis if available, otherwise fall back to step index
        if paragraph_ids is not None and len(paragraph_ids) == n_steps:
            x = np.array([int(pid) for pid in paragraph_ids])
            ax.set_xlabel('Paragraph Number', fontsize=18)
        else:
            x = np.arange(n_steps)
            ax.set_xlabel('Step', fontsize=12)
            if paragraph_ids is not None:
                print(f"Warning: ids length ({len(paragraph_ids)}) != n_steps ({n_steps}) for {story_name}, using step index")

        # Plot human reader with confidence intervals (only if human data available)
        if human_dist is not None and len(human_dist) == n_steps:
            human_true = human_dist[:, true_idx]
            human_ci_lower = [c[0][true_idx] for c in human_ci]
            human_ci_upper = [c[1][true_idx] for c in human_ci]
            ax.plot(x, human_true, 'o-', label='Actual Reader', linewidth=2, markersize=8)
            ax.fill_between(x, human_ci_lower, human_ci_upper, alpha=0.3, label='Actual Reader 68% CI')

        # Plot naive reader (no confidence intervals - single run)
        if naive_dist is not None and len(naive_dist) == n_steps:
            naive_true = naive_dist[:, true_idx]
            ax.plot(x, naive_true, 's-', label='Gullible Reader', linewidth=2, markersize=8)

        # Plot know-it-all reader with confidence intervals for artificial stories
        if is_artificial and sampling_dist is not None and len(sampling_dist) == n_steps:
            sampling_true = sampling_dist[:, true_idx]
            ax.plot(x, sampling_true, '^-', label='Know-it-all Reader', linewidth=2, markersize=8)
            if sampling_ci is not None and len(sampling_ci) == n_steps:
                sampling_ci_lower = [c[0][true_idx] for c in sampling_ci]
                sampling_ci_upper = [c[1][true_idx] for c in sampling_ci]
                ax.fill_between(x, sampling_ci_lower, sampling_ci_upper, alpha=0.2,
                                label='Know-it-all Reader 68% CI')

        ax.set_ylabel('Probability of True Culprit', fontsize=18)
        if self.naive_reader_mode != "default":
            title = f'{model_display} - Story {story_id} [{self.naive_reader_mode}]'
        else:
            title = f'{model_name} - Story {story_id}'
        if num_responses is not None:
            title += f' - {num_responses} human responses'
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.legend(fontsize=16)
        ax.tick_params(axis='x', labelsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # Add FP_UB / S / C annotation using default naive baseline
        if naive_dist is not None and len(naive_dist) == n_steps and sampling_dist is not None and len(sampling_dist) == n_steps:
            _naive_true = naive_dist[:, true_idx]
            _sampling_true = sampling_dist[:, true_idx]
            fp_ub = float(np.mean(_sampling_true - _naive_true)) * 25
            s_val = float(np.mean(_naive_true)) * 25
            c_val = float(np.mean(_sampling_true)) * 25
            annotation = f"FP_UB={fp_ub:.2f}  S={s_val:.2f}  C={c_val:.2f}"
            ax.text(0.02, 0.02, annotation, transform=ax.transAxes,
                    fontsize=14, verticalalignment='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        plt.tight_layout()

        if save_path:
            self._savefig(save_path)

        plt.show()


    # =========================================================================
    # Bar plot with experienced reader (NEW)
    # =========================================================================

    def plot_fairplay_bar_with_experienced(self, save_path=None, experienced_settings=None, with_human=True):
        """Create a bar plot comparing Human, LLM (know-it-all), and Experienced Reader
        fair play scores (aggregate across all stories).

        Args:
            save_path: Path to save the plot.
            experienced_settings: List of experienced reader settings to include
                (0–5; see ``plot_fairplay_comparison`` docstring).  Default ``[1]``.
            with_human: If True (default), include human fair play and restrict to stories
                with human data.  If False, omit human and use all LLM stories; the
                experienced-reader is similarly compared over all available stories.
        """
        if experienced_settings is None:
            experienced_settings = [1]

        # Pre-fetch experienced reader score dicts (llm_key -> fp_score)
        exp_score_dicts = {}
        for setting in experienced_settings:
            exp_score_dicts[setting] = self.get_experienced_reader_fairplay_scores(setting=setting)

        human_fp_all = []
        llm_fp_all = []
        exp_fp_all = {s: [] for s in experienced_settings}

        if with_human:
            # ---- WITH HUMAN: only stories that have human data ----
            for story_name in self.human_data.keys():
                human_dist, _, _ = self.get_human_distribution(story_name)
                if human_dist is None:
                    continue

                naive_dist, true_idx = self.get_naive_reader_probs(story_name)
                if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                    continue

                llm_key, _ = self._resolve_llm_key(story_name)
                sampling_dist, _, _, _, _ = self.get_llm_distributions(*llm_key)

                if sampling_dist is None or len(sampling_dist) != len(human_dist):
                    continue

                # Check that ALL experienced settings have data for this story
                exp_fps_this_story = {}
                all_present = True
                for setting in experienced_settings:
                    if llm_key in exp_score_dicts[setting]:
                        exp_fps_this_story[setting] = exp_score_dicts[setting][llm_key]
                    else:
                        all_present = False
                        break

                if not all_present:
                    continue

                human_true = human_dist[:, true_idx]
                naive_true = naive_dist[:, true_idx]
                sampling_true = sampling_dist[:, true_idx]

                human_fp_all.append(np.mean(human_true - naive_true) * 25)
                llm_fp_all.append(np.mean(sampling_true - naive_true) * 25)
                for setting in experienced_settings:
                    exp_fp_all[setting].append(exp_fps_this_story[setting])

        else:
            # ---- WITHOUT HUMAN: all LLM stories ----
            # Collect the union of keys across all requested experienced settings
            for key in self.llm_data:
                model_name, story_type, story_id = key

                sampling_dist, naive_dist, true_idx, _, _ = self.get_llm_distributions(
                    model_name, story_type, story_id
                )
                if naive_dist is None or sampling_dist is None or true_idx is None:
                    continue
                if len(sampling_dist) != len(naive_dist):
                    continue

                # Check that ALL experienced settings have data for this story
                exp_fps_this_story = {}
                all_present = True
                for setting in experienced_settings:
                    if key in exp_score_dicts[setting]:
                        exp_fps_this_story[setting] = exp_score_dicts[setting][key]
                    else:
                        all_present = False
                        break

                if not all_present:
                    continue

                naive_true = naive_dist[:, true_idx]
                sampling_true = sampling_dist[:, true_idx]
                llm_fp_all.append(np.mean(sampling_true - naive_true) * 25)
                for setting in experienced_settings:
                    exp_fp_all[setting].append(exp_fps_this_story[setting])

        n_stories = len(llm_fp_all)
        if n_stories == 0:
            print("No stories with scores for all requested types.")
            return

        # Build bars
        categories = []
        means = []
        stds = []
        colors_bar = []

        if with_human:
            categories.append('Human Reader')
            means.append(np.mean(human_fp_all))
            stds.append(np.std(human_fp_all))
            colors_bar.append('lightcoral')

        categories.append('Know-it-all (LLM)')
        means.append(np.mean(llm_fp_all))
        stds.append(np.std(llm_fp_all))
        colors_bar.append('lightblue')

        for setting in experienced_settings:
            scores = exp_fp_all[setting]
            short = EXPERIENCED_READER_SHORT_LABELS[setting]
            categories.append(f'Experienced\n({short})')
            means.append(np.mean(scores))
            stds.append(np.std(scores))
            colors_bar.append(EXPERIENCED_READER_COLORS.get(setting, 'lightyellow'))

        fig, ax = plt.subplots(figsize=(max(7, len(categories) * 2), 6))
        x = np.arange(len(categories))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_bar,
                      edgecolor='black', linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel('Fair Play Score (×L)', fontsize=12)
        scope = f'n={n_stories} stories w/ human data' if with_human else f'n={n_stories} all stories'
        ax.set_title(f'Fair Play Comparison (Aggregate, {scope}) — {self.naive_reader_mode}',
                     fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.grid(axis='y', alpha=0.3)

        # Annotate bar values
        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            self._savefig(save_path)

        plt.show()

        # Print summary
        print(f"\nFair Play Aggregate Summary ({scope}):")
        if with_human:
            print(f"  Human:       mean={np.mean(human_fp_all):.3f} ± {np.std(human_fp_all):.3f}")
        print(f"  Know-it-all: mean={np.mean(llm_fp_all):.3f} ± {np.std(llm_fp_all):.3f}")
        for setting in experienced_settings:
            scores = exp_fp_all[setting]
            short = EXPERIENCED_READER_SHORT_LABELS[setting]
            print(f"  Experienced ({short}): mean={np.mean(scores):.3f} ± {np.std(scores):.3f}")

    def plot_coherence_bar(self, save_path=None, experienced_settings=None):
        """Bar plot of coherence (mean prob on true culprit × 25) for five reader types.

        Bars (left → right):
          1. Gullible (naive reader)
          2. Human reader
          3–4. Experienced reader settings (default: 5-prev, 0-shot)
          5. Know-it-all (LLM sampling)

        Only stories with data for all five reader types are included.

        Args:
            save_path: Path to save the plot (.png; PDF also saved if save_pdf is True).
            experienced_settings: List of experienced reader settings to include.
                Default [5, 0] (5-prev then 0-shot).
        """
        if experienced_settings is None:
            experienced_settings = [5, 0]

        naive_coh_all   = []
        human_coh_all   = []
        exp_coh_all     = {s: [] for s in experienced_settings}
        kia_coh_all     = []

        for story_name in self.human_data.keys():
            human_dist, _, _ = self.get_human_distribution(story_name)
            if human_dist is None:
                continue

            naive_dist, true_idx = self.get_naive_reader_probs(story_name)
            if naive_dist is None or true_idx is None or len(naive_dist) != len(human_dist):
                continue

            llm_key, _ = self._resolve_llm_key(story_name)
            sampling_dist, _, _, _, _ = self.get_llm_distributions(*llm_key)
            if sampling_dist is None or len(sampling_dist) != len(human_dist):
                continue

            # Compute experienced reader coherence inline using the same true_idx
            exp_vals = {}
            all_present = True
            for setting in experienced_settings:
                exp_dist, _ = self.get_experienced_reader_distributions(*llm_key, setting=setting)
                if exp_dist is None or len(exp_dist) != len(naive_dist):
                    all_present = False
                    break
                exp_vals[setting] = float(np.mean(exp_dist[:, true_idx])) * 25
            if not all_present:
                continue

            naive_true    = naive_dist[:, true_idx]
            human_true    = human_dist[:, true_idx]
            sampling_true = sampling_dist[:, true_idx]

            naive_coh_all.append(float(np.mean(naive_true))    * 25)
            human_coh_all.append(float(np.mean(human_true))    * 25)
            kia_coh_all.append(  float(np.mean(sampling_true)) * 25)
            for setting in experienced_settings:
                exp_coh_all[setting].append(exp_vals[setting])

        n_stories = len(kia_coh_all)
        if n_stories == 0:
            print("No stories with coherence data for all reader types.")
            return

        # Build bar entries: (label, scores, color)
        bar_entries = [
            ('Gullible\n(naive)', naive_coh_all, 'lightgrey'),
            ('Human\nreader',     human_coh_all,  'lightcoral'),
        ]
        for setting in experienced_settings:
            short = EXPERIENCED_READER_SHORT_LABELS[setting]
            bar_entries.append((
                f'Experienced\n({short})',
                exp_coh_all[setting],
                EXPERIENCED_READER_COLORS.get(setting, 'lightyellow'),
            ))
        bar_entries.append(('Know-it-all\n(LLM)', kia_coh_all, 'lightblue'))

        categories  = [e[0] for e in bar_entries]
        means       = [np.mean(e[1]) for e in bar_entries]
        stds        = [np.std(e[1])  for e in bar_entries]
        colors_bar  = [e[2]          for e in bar_entries]

        fig, ax = plt.subplots(figsize=(max(7, len(categories) * 2), 6))
        x    = np.arange(len(categories))
        bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors_bar,
                      edgecolor='black', linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylabel('Coherence  (mean P(true culprit) × 25)', fontsize=12)
        ax.set_title(
            f'Coherence Comparison — n={n_stories} stories with human data\n'
            f'(naive mode: {self.naive_reader_mode})',
            fontsize=13, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.grid(axis='y', alpha=0.3)

        for bar, mean_val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(stds) * 0.05,
                    f'{mean_val:.2f}', ha='center', va='bottom', fontsize=10)

        plt.tight_layout()

        if save_path:
            self._savefig(save_path)

        plt.show()

        print(f"\nCoherence Summary (n={n_stories} stories):")
        for label, scores, _ in bar_entries:
            label_short = label.replace('\n', ' ')
            print(f"  {label_short:<25}  mean={np.mean(scores):.3f} ± {np.std(scores):.3f}")

    # -------------------------------------------------------------------------
    # Shared scatter helpers
    # -------------------------------------------------------------------------

    _FAMILY_COLOR = {
        'gemini15': '#FFD580',
        'gemini25': '#FF9F40',
        'gemini3':  '#FF6B6B',
        'llama':    '#7BC8F6',
        'gpt':      '#95E08A',
        'other':    '#C8C8C8',
    }
    _FAMILY_LABEL = {
        'gemini15': 'Gemini 1.5',
        'gemini25': 'Gemini 2.5',
        'gemini3':  'Gemini 3',
        'llama':    'Llama',
        'gpt':      'GPT',
        'other':    'Other',
    }

    @staticmethod
    def _model_family(model_key):
        if 'gemini-1.5' in model_key: return 'gemini15'
        if 'gemini-2.5' in model_key: return 'gemini25'
        if 'gemini-3'   in model_key: return 'gemini3'
        if 'Llama'      in model_key: return 'llama'
        if 'gpt'        in model_key: return 'gpt'
        return 'other'

    def _collect_story_metrics(self, min_nsamp=1, skip_real=True):
        """Collect per-story metrics for scatter plots.

        Returns a dict with lists: model, surprise, coherence_llm, coherence_human,
        fp_llm, fp_human.  Human columns are np.nan when no human data exists.
        """
        story_seen = {}
        for key, value in self.llm_data.items():
            model_name, story_type, story_id = key
            nsamp = int(value['params'].get('nsamp', '5'))
            if nsamp < min_nsamp:
                continue
            is_real = story_type in ['poirot', 'sherlock']
            if skip_real and is_real:
                continue
            story_key = f"{model_name}_{story_id}"
            if story_key not in story_seen or nsamp == 20:
                story_seen[story_key] = (key, model_name)

        data = {k: [] for k in ('model', 'surprise', 'coherence_llm', 'coherence_human',
                                'fp_llm', 'fp_human')}
        for story_key, (llm_key, model_name) in story_seen.items():
            _, story_type, story_id = llm_key
            sampling_dist, naive_dist, true_idx, _, _ = self.get_llm_distributions(
                model_name, story_type, story_id
            )
            if sampling_dist is None or naive_dist is None or true_idx is None:
                continue
            if len(sampling_dist) != len(naive_dist):
                continue
            naive_true     = naive_dist[:, true_idx]
            sampling_true  = sampling_dist[:, true_idx]
            data['model'].append(model_name)
            data['surprise'].append((1 - np.mean(naive_true)) * 25)
            data['coherence_llm'].append(np.mean(sampling_true) * 25)
            data['fp_llm'].append(np.mean(sampling_true - naive_true) * 25)

            human_group = model_name if 'gemini-2.5-flash' not in model_name else 'gemini-2.5-flash'
            human_dist, _, _ = self.get_human_distribution(f"{human_group} {story_id}")
            if human_dist is not None and len(human_dist) == len(naive_dist):
                human_true = human_dist[:, true_idx]
                data['coherence_human'].append(np.mean(human_true) * 25)
                data['fp_human'].append(np.mean(human_true - naive_true) * 25)
            else:
                data['coherence_human'].append(np.nan)
                data['fp_human'].append(np.nan)

        return data

    @staticmethod
    def _add_regression(ax, xs, ys, color='crimson', label=True):
        """Add an OLS regression line + 95% CI band to ax."""
        from scipy import stats as _stats
        xs_arr, ys_arr = np.array(xs, dtype=float), np.array(ys, dtype=float)
        n = len(xs_arr)
        if n < 3:
            return
        slope, intercept, r, p, _ = _stats.linregress(xs_arr, ys_arr)
        x_line = np.linspace(xs_arr.min(), xs_arr.max(), 200)
        y_line = slope * x_line + intercept
        x_mean = xs_arr.mean()
        sxx = np.sum((xs_arr - x_mean) ** 2)
        s_res = np.sqrt(np.sum((ys_arr - (slope * xs_arr + intercept)) ** 2) / max(n - 2, 1))
        se_line = s_res * np.sqrt(1 / n + (x_line - x_mean) ** 2 / (sxx + 1e-12))
        t_crit = _stats.t.ppf(0.975, df=max(n - 2, 1))
        lbl = f'OLS r={r:.2f}' if label else None
        ax.plot(x_line, y_line, color=color, linewidth=1.3, alpha=0.85, zorder=4, label=lbl)
        # ax.fill_between(x_line, y_line - t_crit * se_line, y_line + t_crit * se_line,
        #                 color=color, alpha=0.12, zorder=2)

    def plot_metric_scatter(self, x_col, y_col, x_label, y_label, title,
                            save_path=None, min_nsamp=1):
        """Generic per-story scatter plot for any two collected metrics.

        Points are colored by model family.  Only stories with non-NaN values
        for both columns are shown.
        """
        data = self._collect_story_metrics(min_nsamp=min_nsamp)
        n_stories = len(data['model'])

        xs, ys, colors, families_seen = [], [], [], set()
        for i in range(n_stories):
            x = data[x_col][i]
            y = data[y_col][i]
            if np.isnan(x) or np.isnan(y):
                continue
            fam = self._model_family(data['model'][i])
            xs.append(x)
            ys.append(y)
            colors.append(self._FAMILY_COLOR[fam])
            families_seen.add(fam)

        from scipy import stats as _stats
        rho, p = _stats.spearmanr(xs, ys)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xs, ys, c=colors, s=60, edgecolors='k', linewidths=0.4, alpha=0.85, zorder=3)
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)

        # Diagonal only when axes are commensurate (same metric scale)
        if x_col.split('_')[0] == y_col.split('_')[0] or \
                set([x_col, y_col]) <= {'coherence_llm', 'coherence_human',
                                         'fp_llm', 'fp_human'}:
            lo = min(min(xs), min(ys)) - 0.5
            hi = max(max(xs), max(ys)) + 0.5
            ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, alpha=0.4)

        self._add_regression(ax, xs, ys)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(f"{title}\nSpearman ρ={rho:.3f} {sig}  (n={len(xs)})", fontsize=11)

        family_order = ['gemini15', 'gemini25', 'gemini3', 'llama', 'gpt', 'other']
        legend_patches = [
            Patch(facecolor=self._FAMILY_COLOR[f], edgecolor='k', linewidth=0.4,
                  label=self._FAMILY_LABEL[f])
            for f in family_order if f in families_seen
        ]
        ax.legend(handles=legend_patches, fontsize=9, title='Model family',
                  title_fontsize=9, loc='best')

        plt.tight_layout()
        if save_path:
            self._savefig(save_path)
        plt.show()
        print(f"{title}: n={len(xs)}, Spearman ρ={rho:.3f} {sig}")

    def plot_fp_scatter(self, save_path=None, min_nsamp=1):
        """Scatter plot of Human FP (x) vs LLM FP (y), one point per story, colored by model family.

        Args:
            save_path: If given, save the figure to this path (PNG + PDF if save_pdf=True).
            min_nsamp: Minimum nsamp required; stories with fewer samples are excluded.
        """
        FAMILY_COLOR = {
            'gemini15': '#FFD580',
            'gemini25': '#FF9F40',
            'gemini3':  '#FF6B6B',
            'llama':    '#7BC8F6',
            'gpt':      '#95E08A',
            'other':    '#C8C8C8',
        }
        FAMILY_LABEL = {
            'gemini15': 'Gemini 1.5',
            'gemini25': 'Gemini 2.5',
            'gemini3':  'Gemini 3',
            'llama':    'Llama',
            'gpt':      'GPT',
            'other':    'Other',
        }

        def _family(model_key):
            if 'gemini-1.5' in model_key: return 'gemini15'
            if 'gemini-2.5' in model_key: return 'gemini25'
            if 'gemini-3'   in model_key: return 'gemini3'
            if 'Llama'      in model_key: return 'llama'
            if 'gpt'        in model_key: return 'gpt'
            return 'other'

        # Collect per-story FP values (deduplicate, prefer nsamp=20)
        story_seen = {}
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

        xs, ys, colors, families_seen = [], [], [], set()
        for story_key, (llm_key, group, is_real) in story_seen.items():
            if is_real:
                continue
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
            fp_llm = np.mean(sampling_true - naive_true) * 25

            # Look up human data
            human_group = group if 'gemini-2.5-flash' not in group else 'gemini-2.5-flash'
            human_story_name = f"{human_group} {story_id}"
            human_dist, _, _ = self.get_human_distribution(human_story_name)
            if human_dist is None or len(human_dist) != len(naive_dist):
                continue
            human_true = human_dist[:, true_idx]
            fp_human = np.mean(human_true - naive_true) * 25

            fam = _family(group)
            xs.append(fp_human)
            ys.append(fp_llm)
            colors.append(FAMILY_COLOR[fam])
            families_seen.add(fam)

        fig, ax = plt.subplots(figsize=(5, 5))

        ax.scatter(xs, ys, c=colors, s=60, edgecolors='k', linewidths=0.4, alpha=0.85, zorder=3)

        # Diagonal reference line y = x
        all_vals = xs + ys
        lo, hi = min(all_vals) - 0.2, max(all_vals) + 0.2
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, alpha=0.5, label='y = x')
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)

        self._add_regression(ax, xs, ys)
        ax.set_xlabel('Human Fair Play (×25)', fontsize=14)
        ax.set_ylabel('LLM Fair Play (×25)', fontsize=14)
        ax.set_title('Human FP vs. LLM FP (per story)', fontsize=15)

        # Legend: one entry per family present in the data
        family_order = ['gemini15', 'gemini25', 'gemini3', 'llama', 'gpt', 'other']
        legend_patches = [
            Patch(facecolor=FAMILY_COLOR[f], edgecolor='k', linewidth=0.4,
                  label=FAMILY_LABEL[f])
            for f in family_order if f in families_seen
        ]
        ax.legend(handles=legend_patches, fontsize=11, title='Model family',
                  title_fontsize=11, loc='upper left')

        plt.tight_layout()
        if save_path:
            self._savefig(save_path)
        plt.show()
        print(f"FP scatter: n={len(xs)} stories")

    def plot_model_quality_scatter(self, save_path=None, min_nsamp=1):
        """Scatter of per-model mean Surprise (x) vs mean Coherence LLM (y), one point per model.

        Directly tests the quality hypothesis: if model quality drives both metrics, better
        models should appear in the top-right quadrant (high S, high C).
        """
        FAMILY_COLOR = {
            'gemini15': '#FFD580',
            'gemini25': '#FF9F40',
            'gemini3':  '#FF6B6B',
            'llama':    '#7BC8F6',
            'gpt':      '#95E08A',
            'other':    '#C8C8C8',
        }
        FAMILY_LABEL = {
            'gemini15': 'Gemini 1.5',
            'gemini25': 'Gemini 2.5',
            'gemini3':  'Gemini 3',
            'llama':    'Llama',
            'gpt':      'GPT',
            'other':    'Other',
        }

        def _family(model_key):
            if 'gemini-1.5' in model_key: return 'gemini15'
            if 'gemini-2.5' in model_key: return 'gemini25'
            if 'gemini-3'   in model_key: return 'gemini3'
            if 'Llama'      in model_key: return 'llama'
            if 'gpt'        in model_key: return 'gpt'
            return 'other'

        # Collect per-story data (artificial only, prefer nsamp=20)
        story_seen = {}
        for key, value in self.llm_data.items():
            model_name, story_type, story_id = key
            nsamp = int(value['params'].get('nsamp', '5'))
            if nsamp < min_nsamp:
                continue
            if story_type in ['poirot', 'sherlock']:
                continue
            story_key = f"{model_name}_{story_id}"
            if story_key not in story_seen or nsamp == 20:
                story_seen[story_key] = (key, model_name)

        model_records = {}  # model_name -> list of (surprise, coherence_llm)
        for story_key, (llm_key, model_name) in story_seen.items():
            _, story_type, story_id = llm_key
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
            coherence = np.mean(sampling_true) * 25
            model_records.setdefault(model_name, []).append((surprise, coherence))

        xs, ys, colors, labels_list, families_seen = [], [], [], [], set()
        for model_name, pairs in model_records.items():
            arr = np.array(pairs)
            xs.append(np.mean(arr[:, 0]))
            ys.append(np.mean(arr[:, 1]))
            fam = _family(model_name)
            colors.append(FAMILY_COLOR[fam])
            families_seen.add(fam)
            labels_list.append(display_names.get(model_name, model_name))

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xs, ys, c=colors, s=90, edgecolors='k', linewidths=0.5, zorder=3)

        # Label each point with the model short name
        for x, y, lbl in zip(xs, ys, labels_list):
            ax.annotate(lbl, (x, y), textcoords='offset points', xytext=(5, 4),
                        fontsize=7.5, alpha=0.9)

        ax.axhline(np.mean(ys), color='gray', linewidth=0.6, linestyle='--', alpha=0.5)
        ax.axvline(np.mean(xs), color='gray', linewidth=0.6, linestyle='--', alpha=0.5)

        self._add_regression(ax, xs, ys)
        ax.set_xlabel('Mean Surprise (×25)', fontsize=12)
        ax.set_ylabel('Mean Coherence LLM (×25)', fontsize=12)
        ax.set_title('Between-model: Mean Surprise vs. Mean Coherence\n'
                     '(quality hypothesis: positive correlation expected)', fontsize=11)

        family_order = ['gemini15', 'gemini25', 'gemini3', 'llama', 'gpt', 'other']
        legend_patches = [
            Patch(facecolor=FAMILY_COLOR[f], edgecolor='k', linewidth=0.4,
                  label=FAMILY_LABEL[f])
            for f in family_order if f in families_seen
        ]
        ax.legend(handles=legend_patches, fontsize=9, title='Model family',
                  title_fontsize=9, loc='best')

        plt.tight_layout()
        if save_path:
            self._savefig(save_path)
        plt.show()
        print(f"Model quality scatter: n={len(xs)} models")

    def plot_fp_gt1_scatter(self, save_path=None, min_nsamp=1):
        """Per-model scatter: fraction of stories with Human FP > 1 (x) vs LLM FP > 1 (y).

        Each point is one writing model.  Colored by model family.
        """
        FAMILY_COLOR = {
            'gemini15': '#FFD580', 'gemini25': '#FF9F40', 'gemini3': '#FF6B6B',
            'llama': '#7BC8F6', 'gpt': '#95E08A', 'other': '#C8C8C8',
        }
        FAMILY_LABEL = {
            'gemini15': 'Gemini 1.5', 'gemini25': 'Gemini 2.5', 'gemini3': 'Gemini 3',
            'llama': 'Llama', 'gpt': 'GPT', 'other': 'Other',
        }

        def _family(m):
            if 'gemini-1.5' in m: return 'gemini15'
            if 'gemini-2.5' in m: return 'gemini25'
            if 'gemini-3'   in m: return 'gemini3'
            if 'Llama'      in m: return 'llama'
            if 'gpt'        in m: return 'gpt'
            return 'other'

        # Collect per-story human FP and LLM FP (artificial only, same logic as plot_fp_scatter)
        story_seen = {}
        for key, value in self.llm_data.items():
            model_name, story_type, story_id = key
            nsamp = int(value['params'].get('nsamp', '5'))
            if nsamp < min_nsamp or story_type in ['poirot', 'sherlock']:
                continue
            story_key = f"{model_name}_{story_id}"
            if story_key not in story_seen or nsamp == 20:
                story_seen[story_key] = (key, model_name)

        model_records = {}  # model_name -> list of (fp_human, fp_llm)
        for story_key, (llm_key, model_name) in story_seen.items():
            mn, st, sid = llm_key
            sampling_dist, naive_dist, true_idx, _, _ = self.get_llm_distributions(mn, st, sid)
            if sampling_dist is None or naive_dist is None or true_idx is None:
                continue
            if len(sampling_dist) != len(naive_dist):
                continue
            naive_true = naive_dist[:, true_idx]
            sampling_true = sampling_dist[:, true_idx]
            fp_llm = np.mean(sampling_true - naive_true) * 25

            human_group = model_name if 'gemini-2.5-flash' not in model_name else 'gemini-2.5-flash'
            human_dist, _, _ = self.get_human_distribution(f"{human_group} {sid}")
            if human_dist is None or len(human_dist) != len(naive_dist):
                continue
            fp_human = np.mean(human_dist[:, true_idx] - naive_true) * 25

            model_records.setdefault(model_name, []).append((fp_human, fp_llm))

        xs, ys, colors, labels_list, families_seen = [], [], [], [], set()
        for model_name, pairs in model_records.items():
            arr = np.array(pairs)
            xs.append(np.mean(arr[:, 0] > 1))
            ys.append(np.mean(arr[:, 1] > 1))
            fam = _family(model_name)
            colors.append(FAMILY_COLOR[fam])
            families_seen.add(fam)
            labels_list.append(display_names.get(model_name, model_name))

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(xs, ys, c=colors, s=90, edgecolors='k', linewidths=0.5, alpha=0.85, zorder=3)
        for x, y, lbl in zip(xs, ys, labels_list):
            ax.annotate(lbl, (x, y), textcoords='offset points', xytext=(5, 3),
                        fontsize=7.5, alpha=0.9)

        lo = min(min(xs), min(ys)) - 0.05
        hi = max(max(xs), max(ys)) + 0.05
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, alpha=0.45, label='y = x')
        ax.axhline(0, color='gray', linewidth=0.5, alpha=0.4)
        ax.axvline(0, color='gray', linewidth=0.5, alpha=0.4)

        self._add_regression(ax, xs, ys)

        ax.set_xlabel('Fraction of stories: Human FP > 1', fontsize=12)
        ax.set_ylabel('Fraction of stories: LLM FP > 1', fontsize=12)
        ax.set_title('Human vs. LLM: fraction of stories with FP > 1\n(per model)', fontsize=11)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        family_order = ['gemini15', 'gemini25', 'gemini3', 'llama', 'gpt', 'other']
        legend_patches = [
            Patch(facecolor=FAMILY_COLOR[f], edgecolor='k', linewidth=0.4, label=FAMILY_LABEL[f])
            for f in family_order if f in families_seen
        ]
        ax.legend(handles=legend_patches, fontsize=9, title='Model family',
                  title_fontsize=9, loc='upper left')

        plt.tight_layout()
        if save_path:
            self._savefig(save_path)
        plt.show()
        print(f"FP>1 scatter: n={len(xs)} models")

    def plot_model_ordering(self, save_path=None):
        """Plot the model quality partial ordering as a Hasse diagram.

        Nodes are grouped by family (Gemini Flash, Gemini Pro, Llama, GPT-4o),
        arranged bottom (worst) to top (best).  Edges point upward.
        Only models that have artificial-story LLM data are shown.
        """
        import networkx as nx

        # (family, version, tier_or_size)
        # gemini: tier 0=flash, 1=pro  — pro > flash at same major version
        # llama:  size in B
        # gpt_4o: 0=mini, 1=full
        _META = {
            'gemini-1.5-flash':       ('gemini', 1.5, 0),
            'gemini-2.5-flash-tb0':   ('gemini', 2.5, 0),
            'gemini-2.5-flash-tb-1':  ('gemini', 2.5, 0),
            'gemini-3-flash-preview':  ('gemini', 3.0, 0),
            'gemini-1.5-pro':          ('gemini', 1.5, 1),
            'gemini-2.5-pro':          ('gemini', 2.5, 1),
            'gemini-3.1-pro-preview':  ('gemini', 3.1, 1),
            'Llama-3.1-70B-Instruct':  ('llama',  3.1, 70),
            'Llama-3.3-70B-Instruct':  ('llama',  3.3, 70),
            'Llama-3.1-8B-Instruct':   ('llama',  3.1,  8),
            'Llama-3.2-3B-Instruct':   ('llama',  3.2,  3),
            'Llama-3.2-1B-Instruct':   ('llama',  3.2,  1),
            'gpt-4o':                  ('gpt_4o', 4.0,  1),
            'gpt-4o-mini':             ('gpt_4o', 4.0,  0),
        }

        def _vgroup(m):
            """Visual group for layout: 'gemini_flash', 'gemini_pro', 'llama', 'gpt_4o'."""
            fa, _, sa = _META[m][0], _META[m][1], _META[m][2]
            if fa == 'gemini':
                return 'gemini_flash' if sa == 0 else 'gemini_pro'
            return fa

        def _cmp(ma, mb):
            fa, va, sa = ma
            fb, vb, sb = mb
            if fa != fb:
                return 0
            if fa == 'gemini':
                maj_a, maj_b = int(va), int(vb)
                tier_a, tier_b = sa, sb
                if maj_a == maj_b:
                    if tier_a != tier_b:
                        return +1 if tier_a > tier_b else -1
                    return +1 if va > vb else (-1 if vb > va else 0)
                else:
                    if tier_a != tier_b:
                        return 0   # conflicting
                    return +1 if maj_a > maj_b else -1
            newer = (va > vb) - (vb > va)
            larger = (sa > sb) - (sb > sa)
            if newer == 0: return larger
            if larger == 0 or newer == larger: return newer
            return 0

        # Only show models that have artificial-story LLM data
        available = {mn for mn, st, _ in self.llm_data if st not in ('poirot', 'sherlock')}
        models = [m for m in _META if m in available]

        # Build full DAG: edge worse → better
        G_full = nx.DiGraph()
        G_full.add_nodes_from(models)
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                rel = _cmp(_META[ma], _META[mb])
                if   rel == +1: G_full.add_edge(mb, ma)
                elif rel == -1: G_full.add_edge(ma, mb)

        # Transitive reduction → Hasse diagram
        G = nx.transitive_reduction(G_full)
        G.add_nodes_from(G_full.nodes())

        # Level = longest path from any source
        levels = {}
        for node in nx.topological_sort(G_full):
            preds = list(G_full.predecessors(node))
            levels[node] = max((levels[p] + 1 for p in preds), default=0)
        max_level = max(levels.values(), default=0)

        # Visual groups determine column x-position and node color
        VGROUP_ORDER  = ['gemini_flash', 'gemini_pro', 'llama', 'gpt_4o']
        VGROUP_COLOR  = {
            'gemini_flash': '#FFD580',
            'gemini_pro':   '#FF9F40',
            'llama':        '#7BC8F6',
            'gpt_4o':       '#95E08A',
        }
        VGROUP_LABEL  = {
            'gemini_flash': 'Gemini Flash',
            'gemini_pro':   'Gemini Pro',
            'llama':        'Llama',
            'gpt_4o':       'GPT-4o',
        }
        X_CENTERS = {'gemini_flash': 0.0, 'gemini_pro': 2, 'llama': 6, 'gpt_4o': 8}
        SPACING   = 1.1

        # Assign (x, y) positions by visual group + level
        pos = {}
        for vg in VGROUP_ORDER:
            vg_nodes = [m for m in models if _vgroup(m) == vg]
            level_groups = {}
            for m in vg_nodes:
                level_groups.setdefault(levels[m], []).append(m)
            xc = X_CENTERS[vg]
            for lv, grp in level_groups.items():
                grp_sorted = sorted(grp)
                n = len(grp_sorted)
                for i, m in enumerate(grp_sorted):
                    pos[m] = (xc + (i - (n - 1) / 2) * SPACING, float(lv))

        # Draw
        fig, ax = plt.subplots(figsize=(14, max(4, max_level * 2 + 2)))

        node_colors = [VGROUP_COLOR[_vgroup(m)] for m in G.nodes()]

        # Separate intra-group and cross-group (flash↔pro) edges for styling
        intra_edges = [(u, v) for u, v in G.edges() if _vgroup(u) == _vgroup(v)]
        cross_edges = [(u, v) for u, v in G.edges() if _vgroup(u) != _vgroup(v)]

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=700, edgecolors='#333333', linewidths=0.8)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=intra_edges,
                               arrows=True, arrowstyle='-|>', arrowsize=18,
                               edge_color='#444444', width=1.4,
                               min_source_margin=18, min_target_margin=18)
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=cross_edges,
                               arrows=True, arrowstyle='-|>', arrowsize=18,
                               edge_color='#AA4444', width=1.2, style='dashed',
                               min_source_margin=18, min_target_margin=18)

        # Labels below each node
        for m, (x, y) in pos.items():
            ax.text(x, y - 0.28, display_names.get(m, m),
                    ha='center', va='top', fontsize=7.5,
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', alpha=0.7, lw=0))

        # Visual-group headers above the top of each column
        for vg in VGROUP_ORDER:
            vg_nodes = [m for m in models if _vgroup(m) == vg]
            if not vg_nodes:
                continue
            xs_vg = [pos[m][0] for m in vg_nodes]
            ax.text(np.mean(xs_vg), max_level + 0.55, VGROUP_LABEL[vg],
                    ha='center', va='bottom', fontsize=10, fontweight='bold',
                    color=VGROUP_COLOR[vg],
                    bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=VGROUP_COLOR[vg], lw=1.2))

        ax.set_title('Model Quality Partial Ordering — Hasse Diagram\n'
                     'Solid: same tier (newer/larger).  Dashed red: Pro > Flash (same major version).',
                     fontsize=11)
        ax.set_ylabel('Quality level', fontsize=10)
        ax.set_yticks(range(max_level + 1))
        ax.tick_params(left=True, bottom=False, labelbottom=False)
        for spine in ('top', 'right', 'bottom'):
            ax.spines[spine].set_visible(False)

        all_x = [x for x, _ in pos.values()]
        ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
        ax.set_ylim(-0.3, max_level + 0.3)

        plt.tight_layout()
        if save_path:
            self._savefig(save_path)
        plt.show()

    def plot_hasse_edge_colored(self, naive_mode=None, save_path=None, horizontal=False):
        """Hasse diagram with multiple parallel arrows per edge.

        Each edge carries N arrows (one per quality metric), drawn side-by-side
        perpendicular to the edge direction.
        Green = 'better' model scores higher (hypothesis supported).
        Red   = 'better' model scores lower (hypothesis contradicted).
        Grey  = tied or data missing.
        Node color encodes model family/tier.
        """
        import networkx as nx
        from matplotlib.lines import Line2D

        model_metrics, _, _META, _cmp = self._compute_model_metrics(naive_mode=naive_mode)

        METRICS = [
            ('surprise',          'Surprise'),
            ('coherence',         'Coherence'),
            ('fp_gt1_rate',       'LLM FP >= 1'),
            ('human_fp_gt1_rate', 'Human FP >= 1'),
            ('cl_def_gt1_rate',   'Unif - Gullible >= 1'),
        ]
        # Dash patterns: gradual progression solid → long → medium → short → dots
        METRIC_LINESTYLES = [
            'solid',          # 1: solid
            (0, (8, 3)),      # 2: long dashes
            (0, (4, 3)),      # 3: medium dashes
            (0, (2, 3)),      # 4: short dashes
            (0, (1, 3)),      # 5: dots
        ]
        METRIC_BADGE_STYLE = 'circle,pad=0.15'   # uniform circle badges
        N_METRICS   = len(METRICS)
        ARROW_STEP  = 0.085   # perpendicular spacing between parallel arrows
        NODE_MARGIN = 0.33   # data-units clearance around node centre
        ARROW_LW    = 5.
        ARROW_HEAD  = 20
        COLOR_SUCCESS = '#2ca02c'   # green
        COLOR_FAILURE = '#d62728'   # red
        COLOR_GREY    = '#cccccc'   # grey (tied / missing)

        def _vgroup(m):
            fa, _, sa = _META[m][0], _META[m][1], _META[m][2]
            return ('gemini_flash' if sa == 0 else 'gemini_pro') if fa == 'gemini' else fa

        available = set(model_metrics.keys())
        models    = [m for m in _META if m in available]

        G_full = nx.DiGraph()
        G_full.add_nodes_from(models)
        for i, ma in enumerate(models):
            for mb in models[i + 1:]:
                rel = _cmp(_META[ma], _META[mb])
                if   rel == +1: G_full.add_edge(mb, ma)
                elif rel == -1: G_full.add_edge(ma, mb)

        G = nx.transitive_reduction(G_full)
        G.add_nodes_from(G_full.nodes())

        levels = {}
        for node in nx.topological_sort(G_full):
            preds = list(G_full.predecessors(node))
            levels[node] = max((levels[p] + 1 for p in preds), default=0)
        max_level = max(levels.values(), default=0)

        VGROUP_ORDER = ['gemini_flash', 'gemini_pro', 'llama', 'gpt_4o']
        VGROUP_COLOR = {
            'gemini_flash': '#FFB830', 'gemini_pro': '#FFB830',
            'llama': '#7BC8F6', 'gpt_4o': '#95E08A',
        }
        VGROUP_LABEL = {
            'gemini_flash': 'Gemini', 'gemini_pro': 'Gemini',
            'llama': 'Llama', 'gpt_4o': 'GPT',
        }
        X_CENTERS = {'gemini_flash': 0.0, 'gemini_pro': 1.5, 'llama': 3.25, 'gpt_4o': 5}
        # In horizontal mode: quality level → x, group → y (GPT on top)
        Y_CENTERS = {'gemini_flash': 0.0, 'gemini_pro': 1., 'llama': 2.25, 'gpt_4o': 3.5}
        SPACING      = 0.9
        VERT_SPACING = 1.65   # distance between quality levels; increase to spread nodes apart

        pos = {}
        for vg in VGROUP_ORDER:
            vg_nodes = [m for m in models if _vgroup(m) == vg]
            level_groups = {}
            for m in vg_nodes:
                level_groups.setdefault(levels[m], []).append(m)
            for lv, grp in level_groups.items():
                grp_s = sorted(grp)
                n = len(grp_s)
                for i, m in enumerate(grp_s):
                    offset = (i - (n - 1) / 2) * SPACING
                    if horizontal:
                        yc = Y_CENTERS[vg]
                        pos[m] = (float(lv) * VERT_SPACING, yc + offset)
                    else:
                        xc = X_CENTERS[vg]
                        pos[m] = (xc + offset, float(lv) * VERT_SPACING)

        node_colors = [VGROUP_COLOR[_vgroup(m)] for m in G.nodes()]

        # Short names: strip family prefix since node color encodes it; wrap TB onto second line
        def _short_name(m):
            dn = display_names.get(m, m)
            for prefix in ('Gemini ', 'Llama ', 'GPT-'):
                if dn.startswith(prefix):
                    dn = dn[len(prefix):]
                    break
            # wrap "TB..." suffix onto second line for readability
            for tb in (' TB-', ' TB'):
                if tb in dn:
                    dn = dn.replace(tb, '\n' + tb.strip(), 1)
                    break
            return dn

        # Edges between gemini_flash and gemini_pro get extra margin (diagonal arrows)
        DIAGONAL_MARGIN = 0.5

        if horizontal:
            _ys = [y for _, y in pos.values()]
            _y_range = max(_ys) - min(_ys)
            fig, ax = plt.subplots(figsize=(max(10, max_level * VERT_SPACING * 2 + 2), max(4, _y_range * 2 + 2)))
        else:
            fig, ax = plt.subplots(figsize=(14, max(5, max_level * VERT_SPACING * 2 + 2)))

        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                               node_size=6000, edgecolors='#333333', linewidths=1.)

        # Model name inside each node
        node_labels = {m: _short_name(m) for m in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, ax=ax,
                                font_size=14, font_weight='bold')

        # Draw N parallel arrows per edge
        offsets = np.linspace(-(N_METRICS - 1) / 2, (N_METRICS - 1) / 2, N_METRICS) * ARROW_STEP - 0.04
        GEN_SUCCESS_OFFSET = (N_METRICS - 1) / 2 * ARROW_STEP + 0.05  # separated from main bundle

        # Pre-compute diagonal in/out degree: DIAGONAL_MARGIN only applies at an endpoint
        # when ≥2 diagonal edges converge there (avoids shortening single-diagonal endpoints)
        def _is_diag(u, v):
            _dx, _dy = pos[v][0] - pos[u][0], pos[v][1] - pos[u][1]
            return (abs(_dy) > 0.3) if horizontal else (abs(_dx) > 0.3)

        # Total out/in degree per node (all edges, not just diagonal)
        total_out = {}
        total_in  = {}
        for u, v in G.edges():
            total_out[u] = total_out.get(u, 0) + 1
            total_in[v]  = total_in.get(v, 0) + 1

        for u, v in G.edges():
            ux, uy = pos[u]
            vx, vy = pos[v]
            dx, dy = vx - ux, vy - uy
            length = np.sqrt(dx**2 + dy**2)
            if length < 1e-6:
                continue
            ex, ey = dx / length, dy / length   # unit vector along edge
            px, py = -ey, ex                     # unit perpendicular

            # Diagonal margin applies only when ≥2 total edges meet at that endpoint
            is_diagonal = _is_diag(u, v)
            if is_diagonal:
                src_margin = DIAGONAL_MARGIN if total_out.get(u, 0) >= 2 else NODE_MARGIN
                tgt_margin = DIAGONAL_MARGIN if total_in.get(v, 0) >= 2 else NODE_MARGIN
            else:
                src_margin = tgt_margin = NODE_MARGIN
            shaft_z, head_z = (1, 2) if is_diagonal else (3, 4)

            # Shorten to clear node circles
            sx, sy = ux + ex * src_margin, uy + ey * src_margin
            tx, ty = vx - ex * tgt_margin, vy - ey * tgt_margin

            for (mkey, _), off, ls, idx in zip(
                    METRICS, offsets, METRIC_LINESTYLES, range(1, N_METRICS + 1)):
                val_v = model_metrics.get(v, {}).get(mkey)
                val_u = model_metrics.get(u, {}).get(mkey)

                if val_v is None or val_u is None:
                    continue                 # missing data: omit entirely
                elif val_v > val_u:
                    arrow_color = COLOR_SUCCESS
                    arrowstyle  = '-|>'     # head at v (better model): success
                elif val_v < val_u:
                    arrow_color = COLOR_FAILURE
                    arrowstyle  = '<|-'     # head at u (worse model): failure
                else:
                    arrow_color = COLOR_GREY
                    arrowstyle  = '-'       # tied: no head

                ox, oy = px * off, py * off
                # Head size in data units (approximate): used to extend shaft flush with head base
                HEAD_OVERSHOOT = 0
                if arrowstyle == '-|>':
                    shaft_end_x = tx + ox + ex * HEAD_OVERSHOOT
                    shaft_end_y = ty + oy + ey * HEAD_OVERSHOOT
                    head_xy = (tx + ox, ty + oy)
                    head_from = (tx + ox - ex * 0.001, ty + oy - ey * 0.001)
                elif arrowstyle == '<|-':
                    shaft_end_x = tx + ox
                    shaft_end_y = ty + oy
                    head_xy = (sx + ox, sy + oy)
                    head_from = (sx + ox + ex * 0.001, sy + oy + ey * 0.001)
                else:
                    shaft_end_x = tx + ox
                    shaft_end_y = ty + oy
                    head_xy = head_from = None
                # Dashed shaft (extended slightly to meet arrowhead base)
                ax.annotate('',
                    xy=(shaft_end_x, shaft_end_y),
                    xytext=(sx + ox, sy + oy),
                    arrowprops=dict(arrowstyle='-', color=arrow_color,
                                    lw=ARROW_LW, linestyle=ls),
                    annotation_clip=False, zorder=shaft_z)
                # Solid arrowhead drawn separately
                if head_xy is not None:
                    ax.annotate('',
                        xy=head_xy, xytext=head_from,
                        arrowprops=dict(arrowstyle='-|>', color=arrow_color,
                                        lw=ARROW_LW, mutation_scale=ARROW_HEAD,
                                        linestyle='solid'),
                        annotation_clip=False, zorder=head_z)

                # Metric number badge at midpoint (all drawn arrows, including tied)
                mid_x = (sx + tx) / 2 + ox
                mid_y = (sy + ty) / 2 + oy
                ax.text(mid_x, mid_y, str(idx), ha='center', va='center',
                        fontsize=10, color=arrow_color, fontweight='bold', zorder=head_z + 1,
                        bbox=dict(boxstyle=METRIC_BADGE_STYLE, fc='white', ec=arrow_color,
                                  alpha=0.92, lw=1.2))

            # Gen-success arrow — black, separated from main bundle, no badge
            val_v_gs = model_metrics.get(v, {}).get('gen_success')
            val_u_gs = model_metrics.get(u, {}).get('gen_success')
            if val_v_gs is not None and val_u_gs is not None:
                ox_gs, oy_gs = px * GEN_SUCCESS_OFFSET, py * GEN_SUCCESS_OFFSET
                gs_src = (sx + ox_gs, sy + oy_gs)
                gs_tgt = (tx + ox_gs, ty + oy_gs)
                if val_v_gs > val_u_gs:
                    gs_head_xy   = gs_tgt
                    gs_head_from = (tx + ox_gs - ex * 0.001, ty + oy_gs - ey * 0.001)
                elif val_v_gs < val_u_gs:
                    gs_head_xy   = gs_src
                    gs_head_from = (sx + ox_gs + ex * 0.001, sy + oy_gs + ey * 0.001)
                else:
                    gs_head_xy = gs_head_from = None
                # Shaft (always source→target, no head)
                ax.annotate('', xy=gs_tgt, xytext=gs_src,
                    arrowprops=dict(arrowstyle='-', color='black',
                                    lw=ARROW_LW, linestyle='solid'),
                    annotation_clip=False, zorder=shaft_z)
                # Arrowhead drawn as a tiny separate arrow
                if gs_head_xy is not None:
                    ax.annotate('', xy=gs_head_xy, xytext=gs_head_from,
                        arrowprops=dict(arrowstyle='-|>', color='black',
                                        lw=ARROW_LW, mutation_scale=ARROW_HEAD,
                                        linestyle='solid'),
                        annotation_clip=False, zorder=head_z)

        # Group headers
        drawn_h_labels = set()
        for vg in VGROUP_ORDER:
            vg_nodes = [m for m in models if _vgroup(m) == vg]
            if not vg_nodes:
                continue
            label = VGROUP_LABEL[vg]
            color = VGROUP_COLOR[vg]
            if horizontal:
                if label in drawn_h_labels:
                    continue
                # collect all nodes sharing this label (e.g. both gemini subgroups)
                same_label_nodes = [m for m in models if VGROUP_LABEL[_vgroup(m)] == label]
                ys_vg = [pos[m][1] for m in same_label_nodes]
                ax.text(-0.7, np.mean(ys_vg), label,
                        ha='right', va='center', fontsize=24, fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=1.4))
                drawn_h_labels.add(label)
            else:
                xs_vg = [pos[m][0] for m in vg_nodes]
                ax.text(np.mean(xs_vg), -0.6, label,
                        ha='center', va='top', fontsize=14, fontweight='bold',
                        color=color,
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec=color, lw=1.2))

        # Legend: metric line widths + badge shapes + success/failure meaning
        legend_elems = [
            Line2D([0], [0], color='#555555', lw=ARROW_LW, linestyle=ls,
                   label=f'{i+1}. {lbl}')
            for i, ((_, lbl), ls) in enumerate(zip(METRICS, METRIC_LINESTYLES))
        ] + [
            Line2D([0], [0], color='black', lw=ARROW_LW, linestyle='solid',
                   label='Gen Success'),
            Line2D([0], [0], color=COLOR_SUCCESS, lw=ARROW_LW,
                   label=('→ (head at right) = supports' if horizontal else '→ (head at top) = supports ordering')),
            Line2D([0], [0], color=COLOR_FAILURE, lw=ARROW_LW,
                   label=('← (head at left) = contradicts' if horizontal else '← (head at bottom) = contradicts')),
            Line2D([0], [0], color=COLOR_GREY, lw=ARROW_LW,
                   label='— (grey, no head) = tied'),
        ]
        ax.legend(handles=legend_elems, fontsize=18, loc='upper right',
                  bbox_to_anchor=(1.0, 0.97),
                  title='Line style = metric', title_fontsize=19, framealpha=0.9)

        mode_str = naive_mode or self.naive_reader_mode
        # ax.set_title(
        #     f'Model Ordering — Parallel Arrows per Edge  (naive mode: {mode_str})\n'
        #     'Line style = metric; green→ = supports; red← = contradicts; number = metric ID',
        #     fontsize=16)
        scaled_ticks = [lv * VERT_SPACING for lv in range(max_level + 1)]
        for spine in ('top', 'right', 'bottom', 'left'):
            ax.spines[spine].set_visible(False)
        if horizontal:
            ax.set_xlabel('Quality level', fontsize=18)
            ax.set_xticks(scaled_ticks)
            ax.set_xticklabels(range(max_level + 1))
            ax.tick_params(bottom=True, left=False, labelleft=False)
            ax.annotate('', xy=(1., 0), xytext=(0., 0),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='black', lw=8.0,
                                        mutation_scale=20),
                        annotation_clip=False)
            all_y = [y for _, y in pos.values()]
            ax.set_xlim(-1.4, max_level * VERT_SPACING + 0.5)  # extra room on left for group labels
            ax.set_ylim(min(all_y) - 0.5, max(all_y) + 0.3)
        else:
            ax.set_ylabel('Quality level', fontsize=14)
            ax.set_yticks(scaled_ticks)
            ax.set_yticklabels(range(max_level + 1))
            ax.tick_params(left=True, bottom=False, labelbottom=False)
            ax.annotate('', xy=(0, 1.), xytext=(0, 0),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color='black', lw=5.0,
                                        mutation_scale=20),
                        annotation_clip=False)
            all_x = [x for x, _ in pos.values()]
            ax.set_xlim(min(all_x) - 0.5, max(all_x) + 0.5)
            ax.set_ylim(-1.0, max_level * VERT_SPACING + 0.3)

        plt.tight_layout()
        if save_path:
            self._savefig(save_path)
        plt.show()

    def plot_model_metrics_heatmap(self, save_path=None):
        """Heatmap of per-model success metrics across all dimensions.

        Rows = models grouped by family, sorted best-first within each group.
        Columns = metrics grouped as: Default mode | Clueless mode | Diff & Other.
        Cell color = z-score (RdYlGn: green=high/good, red=low/bad).
        Cell text  = raw value.
        """
        import networkx as nx
        from config import ignore as ignore_list

        mm_def, _, _META, _cmp = self._compute_model_metrics(naive_mode='default')
        mm_cl,  _, _,     _    = self._compute_model_metrics(naive_mode='clueless')

        # Compute real story (Poirot/Sherlock) metrics and inject into mm_def/mm_cl
        real_story_sids = {}
        for key in self.llm_data:
            _, st, sid = key
            if st in ('poirot', 'sherlock'):
                real_story_sids.setdefault(st, set()).add(sid)

        saved_mode = self.naive_reader_mode
        for story_type, sids in sorted(real_story_sids.items()):
            recs = []
            for sid in sorted(sids):
                story_name = f"{story_type} {sid}"
                if story_name in ignore_list:
                    continue
                self.naive_reader_mode = 'default'
                nd_def, ti_def = self.get_naive_reader_probs(story_name)
                self.naive_reader_mode = 'clueless'
                nd_cl, ti_cl = self.get_naive_reader_probs(story_name)
                self.naive_reader_mode = saved_mode
                if nd_def is None or ti_def is None:
                    continue
                human_dist, _, _ = self.get_human_distribution(story_name)
                fp_human_def = None
                if human_dist is not None and len(human_dist) == len(nd_def):
                    fp_human_def = (np.mean(human_dist[:, ti_def]) - np.mean(nd_def[:, ti_def])) * 25
                fp_human_cl = None
                if nd_cl is not None and ti_cl is not None and human_dist is not None and len(human_dist) == len(nd_cl):
                    fp_human_cl = (np.mean(human_dist[:, ti_cl]) - np.mean(nd_cl[:, ti_cl])) * 25
                cl_def_diff = None
                if nd_cl is not None and ti_cl is not None and len(nd_cl) == len(nd_def):
                    cl_def_diff = (np.mean(nd_cl[:, ti_cl]) - np.mean(nd_def[:, ti_def])) * 25
                recs.append({'fp_human_def': fp_human_def, 'fp_human_cl': fp_human_cl, 'cl_def_diff': cl_def_diff})
            if not recs:
                continue
            fp_h_def = [r['fp_human_def'] for r in recs if r['fp_human_def'] is not None]
            fp_h_cl  = [r['fp_human_cl']  for r in recs if r['fp_human_cl']  is not None]
            cl_ds    = [r['cl_def_diff']   for r in recs if r['cl_def_diff']  is not None]
            base = {'fp_gt1_rate': None, 'gen_success': None, 'no_early_reveal_p1': None,
                    'cl_def_gt1_rate': np.mean([v > 1 for v in cl_ds]) if cl_ds else None,
                    'n': len(recs)}
            mm_def[story_type] = {**base, 'human_fp_gt1_rate': np.mean([v > 1 for v in fp_h_def]) if fp_h_def else None}
            mm_cl[story_type]  = {**base, 'human_fp_gt1_rate': np.mean([v > 1 for v in fp_h_cl])  if fp_h_cl  else None}
        self.naive_reader_mode = saved_mode

        def _vgroup(m):
            if m in ('poirot', 'sherlock'):
                return 'real'
            fa, _, sa = _META[m][0], _META[m][1], _META[m][2]
            return ('gemini_flash' if sa == 0 else 'gemini_pro') if fa == 'gemini' else fa

        available = set(mm_def.keys())
        models_all = [m for m in _META if m in available]

        # Build levels from Hasse diagram for row ordering
        G_full = nx.DiGraph()
        G_full.add_nodes_from(models_all)
        for i, ma in enumerate(models_all):
            for mb in models_all[i + 1:]:
                rel = _cmp(_META[ma], _META[mb])
                if   rel == +1: G_full.add_edge(mb, ma)
                elif rel == -1: G_full.add_edge(ma, mb)
        levels = {}
        for node in nx.topological_sort(G_full):
            preds = list(G_full.predecessors(node))
            levels[node] = max((levels[p] + 1 for p in preds), default=0)

        VGROUP_ORDER = ['gemini_flash', 'gemini_pro', 'llama', 'gpt_4o', 'real']
        VGROUP_LABEL = {
            'gemini_flash': 'Gemini\nFlash', 'gemini_pro': 'Gemini\nPro',
            'llama': 'Llama', 'gpt_4o': 'GPT-4o', 'real': 'Real\nStories',
        }
        sorted_models = []
        for vg in VGROUP_ORDER:
            if vg == 'real':
                grp = sorted([m for m in ('poirot', 'sherlock') if m in available])
            else:
                grp = [m for m in models_all if _vgroup(m) == vg]
                grp.sort(key=lambda m: -levels.get(m, 0))
            sorted_models.extend(grp)

        # Column definitions: (key, header, value_getter(mm_def, mm_cl, mn))
        # Only ratio/rate columns (no continuous means)
        COLUMNS = [
            # Validation
            ('gen_success', 'Gen\nSuccess',      lambda d, c, m: d[m].get('gen_success')),
            # Default mode rates
            ('fpgt1_def',   'LLM FP>1\n(def)',   lambda d, c, m: d[m].get('fp_gt1_rate')),
            ('hgt1_def',    'Hum FP>1\n(def)',   lambda d, c, m: d[m].get('human_fp_gt1_rate')),
            # Clueless mode rates
            ('fpgt1_cl',    'LLM FP>1\n(cl)',    lambda d, c, m: c[m].get('fp_gt1_rate')),
            ('hgt1_cl',     'Hum FP>1\n(cl)',    lambda d, c, m: c[m].get('human_fp_gt1_rate')),
            # Other rates
            ('cl_def_r',    'Cl>Def\nrate',      lambda d, c, m: d[m].get('cl_def_gt1_rate')),
            ('no_er',       'No early\nreveal',  lambda d, c, m: d[m].get('no_early_reveal_p1')),
        ]
        COL_SEPARATORS = [0.5, 2.5, 4.5, 5.5]
        COL_GROUPS = [
            ('Validation', 0, 0), ('Default mode', 1, 2),
            ('Clueless mode', 3, 4), ('Other', 5, 6),
        ]

        n_rows = len(sorted_models)
        n_cols = len(COLUMNS)
        raw = np.full((n_rows, n_cols), np.nan)
        for r, mn in enumerate(sorted_models):
            if mn not in mm_def or mn not in mm_cl:
                continue
            for ci, (_, _, getter) in enumerate(COLUMNS):
                val = getter(mm_def, mm_cl, mn)
                if val is not None:
                    raw[r, ci] = float(val)

        # Z-score per column
        z = np.full_like(raw, np.nan)
        for ci in range(n_cols):
            col = raw[:, ci]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                mu, sd = np.nanmean(col), np.nanstd(col)
                if sd > 1e-10:
                    z[valid, ci] = (col[valid] - mu) / sd

        cmap = plt.cm.RdYlGn.copy()
        cmap.set_bad('#dddddd')

        # Cell size in inches: makes cells visually square-ish
        CELL_W, CELL_H = 1.4, 0.9
        fig_w = n_cols * CELL_W + 3.5
        fig_h = n_rows * CELL_H + 2.5
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(np.ma.masked_invalid(z), cmap=cmap, vmin=-2.5, vmax=2.5, aspect='auto')

        # Cell text
        for r in range(n_rows):
            for ci in range(n_cols):
                v = raw[r, ci]
                if not np.isnan(v):
                    zv = z[r, ci]
                    tc = 'white' if abs(zv) > 1.5 else 'black'
                    ax.text(ci, r, f'{v:.2f}', ha='center', va='center',
                            fontsize=9, color=tc)

        # Axes
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([lbl for _, lbl, _ in COLUMNS], fontsize=9)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels([display_names.get(m, m) for m in sorted_models], fontsize=9)
        ax.xaxis.tick_top()

        # Column group separators and labels
        for xs in COL_SEPARATORS:
            ax.axvline(xs, color='black', linewidth=1.5)
        for lbl, c0, c1 in COL_GROUPS:
            ax.text((c0 + c1) / 2, -1.3, lbl, ha='center', va='bottom',
                    fontsize=9, fontweight='bold', transform=ax.get_xaxis_transform())

        # Row group separators and family labels
        # Use pure axes-fraction transform so tight_layout handles margins correctly
        cumsum = 0
        vg_sizes = [sum(1 for m in sorted_models if _vgroup(m) == vg) for vg in VGROUP_ORDER]
        for vg, sz in zip(VGROUP_ORDER, vg_sizes):
            if sz == 0:
                continue
            mid = cumsum + sz / 2 - 0.5
            y_frac = 1.0 - (mid + 0.5) / n_rows   # convert data row coord → axes fraction
            ax.text(-0.22, y_frac, VGROUP_LABEL[vg], ha='center', va='center',
                    fontsize=9, fontweight='bold', transform=ax.transAxes)
            cumsum += sz
            if cumsum < n_rows:
                ax.axhline(cumsum - 0.5, color='black', linewidth=1.5)

        cb = fig.colorbar(im, ax=ax, pad=0.01, fraction=0.018, shrink=0.85)
        cb.set_label('Z-score', fontsize=9)

        ax.set_title('Model Success Metrics Heatmap  (z-scored per column; higher = better)',
                     fontsize=11, pad=35)
        plt.tight_layout()
        if save_path:
            self._savefig(save_path)
        plt.show()