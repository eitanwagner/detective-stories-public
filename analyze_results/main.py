"""Main entry point for detective story analysis.

Combines the core analyzer with analysis and plotting mixins, then
runs the full analysis pipeline. Toggle steps on/off in the RUN CONFIG
section below.
"""

import os
import config as _config

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PLOTS_DIR = os.path.join(_PROJECT_ROOT, "plots")
from config import model_dict  # _config used below for in-place list extension
from analyzer_core import DetectiveStoryAnalyzer as _CoreAnalyzer
from analysis import AnalysisMixin
from plotting import PlottingMixin


class DetectiveStoryAnalyzer(AnalysisMixin, PlottingMixin, _CoreAnalyzer):
    """Full analyzer combining core logic, analysis methods, and plotting."""
    pass


# =========================================================================
# RUN CONFIG — toggle each step on/off
# =========================================================================

# When True, skip loading human data (Excel) and all analyses that require it.
# Only LLM-only plots run: plot_poirot_vs_sherlock(naive=True) and plot_llm_fairplay_all_stories.
NO_HUMAN_DATA = False
# NO_HUMAN_DATA = True

# New models: gemini-3-flash-preview and gemini-3.1-pro-preview
# When True, appends them to model_names_all (and model_names_with_human if applicable).
# Also switch REAL_STORY_S_MODEL in config.py to use gemini-3-flash-preview for real story sampling.
# INCLUDE_NEW_MODELS = False
INCLUDE_NEW_MODELS = True

# Data loading (always needed)
LOAD_EXPERIENCED_READER = True        # load -es2 and -es3 files from with_prev/

# Fairplay scores (fast: just prints numbers)
RUN_FAIRPLAY_SCORES = True
# RUN_FAIRPLAY_SCORES = False
RUN_EXPERIENCED_READER_SCORES = True
# RUN_EXPERIENCED_READER_SCORES = False

# Know-it-all early reveal: fraction of stories where P(true culprit) > threshold at early paragraphs
RUN_EARLY_REVEAL = True

# Model ordering visualizations
RUN_HASSE_EDGE_COLORED = True      # Hasse diagram with parallel arrows per edge, colored by metric
RUN_MODEL_METRICS_HEATMAP = True   # Heatmap: models × all success metrics (default + clueless)

# Subjective ratings (fast: just prints numbers)
RUN_RATINGS_BY_MODEL = True
RUN_SUBJECTIVE_SUMMARY = True

# Correlations (fast)
RUN_CORRELATION_ANALYSIS = True
RUN_METRIC_CORRELATIONS = True   # story-level: surprise/coherence/FP correlations + mixed effects
METRIC_CORR_MIN_NSAMP = 20       # minimum nsamp required for a story to be included (filters out noisy nsamp=5 stories)
CORRELATION_NAIVE_MODE = "default"  # naive reader mode for measured_surprise; measured_surprise_sn always uses super_naive_report
# CORRELATION_NAIVE_MODE = "clueless"  # naive reader mode for measured_surprise; measured_surprise_sn always uses super_naive_report
# CORRELATION_NAIVE_MODE = "super_naive_report"

# Mixed effects models (SLOW — often fail to converge; disable if not needed)
RUN_MIXED_EFFECTS = False           # Master switch: disables all mixed effects sections
RUN_SUBJECTIVE_MIXED_EFFECTS = False
SHOW_NON_CONVERGED = False          # If False (default), skip printing non-converged models

# Plots
# RUN_POIROT_VS_SHERLOCK = False
RUN_POIROT_VS_SHERLOCK = False
# POIROT_VS_SHERLOCK_SHOW_SAMPLING = True  # If False, only gullible/human curves shown (no LLM sampling overlay)
POIROT_VS_SHERLOCK_SHOW_SAMPLING = False
# Naive mode for poirot-vs-sherlock plot (independent of NAIVE_MODES loop)
POIROT_VS_SHERLOCK_NAIVE_MODE = "default"
# POIROT_VS_SHERLOCK_NAIVE_MODE = "clueless"
RUN_FAIRPLAY_COMPARISON_PLOT = True
RUN_FAIRPLAY_WITH_EXPERIENCED_PLOT = True
RUN_AGGREGATE_BAR_PLOT = True
RUN_LLM_FAIRPLAY_ALL_STORIES_PLOT = False

# FP scatter: human FP (x) vs LLM FP (y), one point per story colored by model family
RUN_FP_SCATTER = True

# Actual fair play: human FP and experienced reader FP vs default naive, per model
RUN_ACTUAL_FAIRPLAY = True
ACTUAL_FAIRPLAY_EXP_SETTING = 0   # 0 = 0-shot experienced reader

# Participant/reader analyses (moderate speed)
RUN_PARTICIPANT_VARIANCE = False
RUN_CURVE_DIFFERENCES = True
RUN_READER_PERFORMANCE = True
RUN_LEARNING_CURVES = True                 # Spearman correlations over story order
RUN_LEARNING_CURVE_MIXED_EFFECTS = False  # Mixed effects on top of the above (SLOW)

# Reading curve plots (one plot per story — many plots)
# RUN_READING_CURVE_PLOTS = False
RUN_READING_CURVE_PLOTS = True
# Which stories to plot: "all", "real" (poirot/sherlock only), "artificial" (LLM-written only)
READING_CURVE_STORY_CATEGORY = "all"
# Naive reader mode to use for reading curve plots (independent of NAIVE_MODES above)
READING_CURVE_NAIVE_MODE = "default"
# READING_CURVE_NAIVE_MODE = "clueless"
# READING_CURVE_NAIVE_MODE = "super_naive_threshold"
READING_CURVES_SAVE_DIR = os.path.join(_PLOTS_DIR, "reading_curves")

# Naive reader modes to iterate over for plots
NAIVE_MODES = ["default"]
# NAIVE_MODES = ["clueless"]
# NAIVE_MODES = ["default", "super_naive_threshold", "super_naive_report"]
# NAIVE_MODES = ["default", "no_distractor", "clueless", "clueless_no_distractor", "super_naive_threshold", "super_naive_report"]

# Reading curve stats: print FPUB + human FP per story (for LaTeX subcaptions)
RUN_READING_CURVE_STATS = True

# Name diversity: entropy of first-name distributions per role from _w_details2.json files
RUN_NAME_DIVERSITY = True

# Model ordering test: sign test for newer/larger = better across 5 metrics
RUN_MODEL_ORDERING_TEST = True
RUN_MODEL_ORDERING_PLOT = True

# Save all plots as PDF in addition to PNG
SAVE_PDF = True

# PNAS figures (Figure 1 two-panel whisker + Figure 2 bump/bar chart)
RUN_PNAS_FIGURES = False
# PNAS_NAIVE_MODE = "super_naive_report"   # baseline for FP_UB / S / C_UB
PNAS_NAIVE_MODE = "default"   # baseline for FP_UB / S / C_UB
# PNAS_NAIVE_MODE = "clueless"   # baseline for FP_UB / S / C_UB
PNAS_EXP_SETTING = 5                        # experienced reader setting (5 = 5-prev)
PNAS_SAVE_PREFIX = os.path.join(_PLOTS_DIR, "pnas")
BUMP_INCLUDE_HUMAN_FP1 = True    # If True, add % Human FP>1 column to fig2b (S/C/FP_UB plot)
# BUMP_INCLUDE_HUMAN_FP3 = False   # If True, add % Human FP>3 column to fig2c
BUMP_INCLUDE_HUMAN_FP3 = True   # If True, add % Human FP>3 column to fig2c

# Revelation point comparison: LLM-annotated (true_culprit_paragraph) vs threshold-based
RUN_REVELATION_COMPARISON = True
STORIES_DIR = os.path.join(_PROJECT_ROOT, "stories") + "/"

# Experienced reader settings to show in plots.
#
# Settings 0-2 come from -es2 files:
#   0 = 0-shot (general instructions only)
#   1 = 9 previous stories given
#   2 = 3 previous stories given
#
# Settings 3-5 come from -es3 files:
#   3 = 0-shot with alternative prompt
#   4 = 2 previous stories given
#   5 = 5 previous stories given
#
# Full ordering by n_prev: 0-shot(0), 0-shot-alt(3), 2-prev(4), 3-prev(2), 5-prev(5), 9-prev(1)
# EXPERIENCED_SETTINGS_FOR_PLOT = [0, 3, 4, 2, 5, 1]
EXPERIENCED_SETTINGS_FOR_PLOT = [0, 5]

# Slow analyses that were originally behind `continue` in the loop
RUN_SUBJECTIVE_RATINGS_IN_LOOP = False
RUN_MIXED_EFFECTS_IN_LOOP = False

# =========================================================================


if __name__ == "__main__":
    # Apply new-model flag: extend model lists in-place so analysis.py/plotting.py see the change
    if INCLUDE_NEW_MODELS:
        _config.model_names_all.extend(_config.model_names_new)
        # Uncomment the line below if human data exists for the new models:
        # _config.model_names_with_human.extend(_config.model_names_new)

    # Initialize analyzer
    analyzer = DetectiveStoryAnalyzer(
        excel_path=os.path.join(_PROJECT_ROOT, "results", "for eval2", "Detective stories - All Responses - w - Extra.xlsx"),
        json_dir=os.path.join(_PROJECT_ROOT, "results", "for eval2") + "/",
        model_dict=model_dict,
    )
    analyzer.save_pdf = SAVE_PDF

    # Load data
    print("Loading LLM data...")
    analyzer.load_llm_data()
    if not NO_HUMAN_DATA:
        print("Loading human data...")
        analyzer.load_human_data()

    if not NO_HUMAN_DATA and LOAD_EXPERIENCED_READER:
        print("Loading experienced reader data...")
        analyzer.load_experienced_reader_data()  # loads -es2 and -es3 from json_dir / "with_prev"
        print("\nChecking experienced-reader vs naive true-culprit-index consistency...")
        analyzer.check_experienced_true_idx()

    # =========================================================================
    # Fairplay scores report
    # =========================================================================
    if RUN_FAIRPLAY_SCORES and not NO_HUMAN_DATA:
        print("\n" + "=" * 60)
        print("PRINTING COMPREHENSIVE FAIRPLAY SCORES")
        print("=" * 60 + "\n")
        analyzer.set_naive_reader_mode(PNAS_NAIVE_MODE)
        analyzer.print_all_fairplay_scores()

    if RUN_EXPERIENCED_READER_SCORES and LOAD_EXPERIENCED_READER and not NO_HUMAN_DATA:
        print("\n" + "=" * 60)
        print("PRINTING EXPERIENCED READER SCORES")
        print("=" * 60 + "\n")
        analyzer.print_experienced_reader_scores()

    if RUN_EARLY_REVEAL:
        analyzer.print_early_reveal_stats()

    # =========================================================================
    # Subjective ratings
    # =========================================================================
    if RUN_RATINGS_BY_MODEL and not NO_HUMAN_DATA:
        print("\n" + "=" * 60)
        print("PRINTING SUBJECTIVE RATINGS BY MODEL")
        print("=" * 60 + "\n")
        analyzer.print_ratings_by_model()

    if RUN_SUBJECTIVE_SUMMARY and not NO_HUMAN_DATA:
        print("\n" + "=" * 60)
        print("PRINTING SUBJECTIVE SCORE SUMMARY")
        print("=" * 60 + "\n")
        analyzer.print_subjective_score_summary()

    # =========================================================================
    # Slow mixed effects / correlation analyses
    # =========================================================================
    if RUN_SUBJECTIVE_MIXED_EFFECTS and not NO_HUMAN_DATA:
        print("\n" + "=" * 60)
        print("RUNNING MIXED EFFECTS ANALYSIS FOR SUBJECTIVE RATINGS")
        print("=" * 60 + "\n")
        mixed_subjective_results, mixed_subjective_data = analyzer.analyze_subjective_mixed_effects()

    if RUN_CORRELATION_ANALYSIS and not NO_HUMAN_DATA:
        print("\n" + "=" * 60)
        print("RUNNING CORRELATION ANALYSIS")
        print("=" * 60 + "\n")
        correlation_results, correlation_data = analyzer.analyze_subjective_correlations(
            skip_non_converged=not SHOW_NON_CONVERGED,
            naive_mode=CORRELATION_NAIVE_MODE,
            exp_surprise_setting=ACTUAL_FAIRPLAY_EXP_SETTING,
            run_mixed_effects=RUN_MIXED_EFFECTS,
        )

    if RUN_METRIC_CORRELATIONS:
        print("\n" + "=" * 60)
        print("RUNNING METRIC CORRELATION ANALYSIS")
        print("=" * 60 + "\n")
        metric_corr_results, metric_corr_df = analyzer.analyze_metric_correlations(
            skip_non_converged=not SHOW_NON_CONVERGED,
            min_nsamp=METRIC_CORR_MIN_NSAMP,
            run_mixed_effects=RUN_MIXED_EFFECTS,
        )

    if RUN_FP_SCATTER and not NO_HUMAN_DATA:
        print("\n" + "=" * 60)
        print("PLOTTING FP SCATTER (Human vs LLM)")
        print("=" * 60 + "\n")
        analyzer.plot_fp_scatter(
            save_path=os.path.join(_PLOTS_DIR, "fp_scatter.png"),
        )
        analyzer.plot_model_quality_scatter(
            save_path=os.path.join(_PLOTS_DIR, "model_quality_scatter.png"),
        )
        analyzer.plot_metric_scatter(
            'coherence_human', 'coherence_llm',
            'Human Coherence (×25)', 'LLM Coherence (×25)',
            'Human vs. LLM Coherence (per story)',
            save_path=os.path.join(_PLOTS_DIR, "coherence_human_vs_llm.png"),
        )
        analyzer.plot_metric_scatter(
            'surprise', 'coherence_llm',
            'Surprise (×25)', 'LLM Coherence (×25)',
            'Surprise vs. LLM Coherence (per story)',
            save_path=os.path.join(_PLOTS_DIR, "surprise_vs_coherence_llm.png"),
        )
        analyzer.plot_metric_scatter(
            'surprise', 'coherence_human',
            'Surprise (×25)', 'Human Coherence (×25)',
            'Surprise vs. Human Coherence (per story)',
            save_path=os.path.join(_PLOTS_DIR, "surprise_vs_coherence_human.png"),
        )
        analyzer.plot_fp_gt1_scatter(
            save_path=os.path.join(_PLOTS_DIR, "fp_gt1_scatter.png"),
        )

    # =========================================================================
    # Plots per naive reader mode
    # =========================================================================
    for mode in NAIVE_MODES:
        print(f"\n{'=' * 60}")
        print(f"ANALYZING WITH NAIVE READER MODE: {mode}")
        print(f"{'=' * 60}\n")

        analyzer.set_naive_reader_mode(mode)

        if RUN_POIROT_VS_SHERLOCK:
            print("\nPlotting Poirot vs Sherlock comparison...")
            analyzer.set_naive_reader_mode(POIROT_VS_SHERLOCK_NAIVE_MODE)
            naive_values = [True] if NO_HUMAN_DATA else [False, True]
            for n in naive_values:
                analyzer.plot_poirot_vs_sherlock(
                    save_path=f"poirot_vs_sherlock_{'naive' if n else 'human'}_argmax.png",
                    naive=n, use_argmax=True, show_sampling=POIROT_VS_SHERLOCK_SHOW_SAMPLING
                )
                analyzer.plot_poirot_vs_sherlock(
                    save_path=f"poirot_vs_sherlock_{'naive' if n else 'human'}_popdist.png",
                    naive=n, use_argmax=False, show_sampling=POIROT_VS_SHERLOCK_SHOW_SAMPLING
                )
            analyzer.set_naive_reader_mode(mode)  # restore loop mode

        if RUN_FAIRPLAY_COMPARISON_PLOT and not NO_HUMAN_DATA:
            print("\nPlotting Fair Play comparison (human vs know-it-all)...")
            analyzer.plot_fairplay_comparison(save_path=f"fairplay_comparison_{mode}.png")

        if RUN_FAIRPLAY_WITH_EXPERIENCED_PLOT and LOAD_EXPERIENCED_READER and not NO_HUMAN_DATA:
            print("\nPlotting Fair Play comparison with experienced reader — WITH human (stories w/ human data)...")
            analyzer.plot_fairplay_comparison(
                save_path=f"fairplay_comparison_{mode}_with_exp_human.png",
                experienced_settings=EXPERIENCED_SETTINGS_FOR_PLOT,
                with_human=True
            )

            print("\nPlotting Fair Play comparison with experienced reader — WITHOUT human (all stories)...")
            analyzer.plot_fairplay_comparison(
                save_path=f"fairplay_comparison_{mode}_with_exp_all.png",
                experienced_settings=EXPERIENCED_SETTINGS_FOR_PLOT,
                with_human=False
            )

        if RUN_AGGREGATE_BAR_PLOT and LOAD_EXPERIENCED_READER and not NO_HUMAN_DATA:
            print("\nPlotting Fair Play bar chart (aggregate) — WITH human (stories w/ human data)...")
            analyzer.plot_fairplay_bar_with_experienced(
                save_path=f"fairplay_bar_with_experienced_{mode}_human.png",
                experienced_settings=EXPERIENCED_SETTINGS_FOR_PLOT,
                with_human=True
            )

            print("\nPlotting Fair Play bar chart (aggregate) — WITHOUT human (all stories)...")
            analyzer.plot_fairplay_bar_with_experienced(
                save_path=f"fairplay_bar_with_experienced_{mode}_all.png",
                experienced_settings=EXPERIENCED_SETTINGS_FOR_PLOT,
                with_human=False
            )

            print("\nPlotting Coherence bar chart (aggregate)...")
            analyzer.plot_coherence_bar(
                save_path=f"coherence_bar_{mode}.png",
                experienced_settings=EXPERIENCED_SETTINGS_FOR_PLOT,
            )

        if RUN_SUBJECTIVE_RATINGS_IN_LOOP and not NO_HUMAN_DATA:
            print("\nAnalyzing subjective ratings...")
            results, df_individual, df_stories = analyzer.analyze_subjective_ratings()

        if RUN_MIXED_EFFECTS_IN_LOOP and not NO_HUMAN_DATA:
            results_mixed, df_individual = analyzer.analyze_with_mixed_effects()
            analyzer.visualize_mixed_effects_results(results_mixed, df_individual)

    # =========================================================================
    # Actual fair play (human + exp reader vs default naive)
    # =========================================================================
    if RUN_ACTUAL_FAIRPLAY and not NO_HUMAN_DATA and LOAD_EXPERIENCED_READER:
        print("\n" + "=" * 60)
        print("ACTUAL FAIR PLAY (Human & Exp Reader vs Default Naive)")
        print("=" * 60 + "\n")
        analyzer.plot_actual_fairplay(
            exp_setting=ACTUAL_FAIRPLAY_EXP_SETTING,
            save_path=f"actual_fairplay_exp{ACTUAL_FAIRPLAY_EXP_SETTING}.png",
        )

    # =========================================================================
    # LLM fairplay for ALL stories
    # =========================================================================
    if RUN_LLM_FAIRPLAY_ALL_STORIES_PLOT:
        print("\n" + "=" * 60)
        print("PLOTTING LLM FAIRPLAY FOR ALL STORIES AND MODELS")
        print("=" * 60 + "\n")
        analyzer.plot_llm_fairplay_all_stories(save_path="llm_fairplay_all_stories.png")

    # =========================================================================
    # Participant variance
    # =========================================================================
    if RUN_PARTICIPANT_VARIANCE:
        print("\n" + "=" * 60)
        print("ANALYZING PARTICIPANT PERFORMANCE VARIANCE")
        print("=" * 60 + "\n")
        variance_df = analyzer.analyze_participant_variance()

    # =========================================================================
    # Original analyses (with default mode)
    # =========================================================================
    analyzer.set_naive_reader_mode("default")

    if RUN_READING_CURVE_PLOTS:
        import os as _os
        print("\nPlotting reading curves...")
        analyzer.set_naive_reader_mode(READING_CURVE_NAIVE_MODE)

        if SAVE_PDF:
            _os.makedirs(READING_CURVES_SAVE_DIR, exist_ok=True)

        def _rc_save_path(sname):
            if not SAVE_PDF:
                return None
            fname = sname.replace(' ', '_') + '.png'
            return _os.path.join(READING_CURVES_SAVE_DIR, fname)

        if NO_HUMAN_DATA:
            # Plot for all available stories from llm_data (no human data required)
            seen = set()
            for key in analyzer.llm_data.keys():
                llm_model_name, story_type, story_id = key
                is_real = story_type in ['poirot', 'sherlock']
                if READING_CURVE_STORY_CATEGORY == "real" and not is_real:
                    continue
                if READING_CURVE_STORY_CATEGORY == "artificial" and is_real:
                    continue
                if is_real:
                    if llm_model_name != _config.REAL_STORY_S_MODEL:
                        continue
                    story_name = f"{story_type} {story_id}"
                else:
                    story_name = f"{llm_model_name} {story_id}"
                if story_name not in seen:
                    seen.add(story_name)
                    analyzer.plot_reading_curves(story_name, save_path=_rc_save_path(story_name))
        else:
            for story_name in list(analyzer.human_data.keys()):
                llm_model_name, story_id = story_name.split()
                is_real = llm_model_name in ['poirot', 'sherlock']
                if READING_CURVE_STORY_CATEGORY == "real" and not is_real:
                    continue
                if READING_CURVE_STORY_CATEGORY == "artificial" and is_real:
                    continue
                analyzer.plot_reading_curves(story_name, save_path=_rc_save_path(story_name))

        analyzer.set_naive_reader_mode("default")

    if RUN_CURVE_DIFFERENCES and not NO_HUMAN_DATA:
        print("\nCalculating curve differences...")
        diff_df = analyzer.calculate_curve_differences()
        print("\nCurve Differences Summary:")
        per_model = diff_df.groupby('model').agg({
            'human_naive_diff': ['mean', 'std'],
            'human_naive_exceeds': 'mean',
            'knowitall_naive_diff': ['mean', 'std']
        })
        print(per_model)

    if RUN_READER_PERFORMANCE and not NO_HUMAN_DATA:
        print("\nAnalyzing reader performance...")
        reader_stats = analyzer.analyze_reader_performance()
        print("\nReader Performance Summary:")
        print(reader_stats.groupby('is_artificial')['relative_performance'].describe())

        user_rel_performance = reader_stats.groupby('user_id')['relative_performance'].agg([
            ('avg_performance', 'mean'),
            ('count', 'size'),
            ('std_performance', 'std')
        ]).reset_index()

        if RUN_LEARNING_CURVES:
            print("\nAnalyzing learning curves...")
            learning_results, learning_df = analyzer.analyze_learning_curve(
                reader_stats, run_mixed_effects=RUN_LEARNING_CURVE_MIXED_EFFECTS,
                skip_non_converged=not SHOW_NON_CONVERGED
            )
            print("\nReal stories:")
            real_data = learning_results['real']
            print(f"Overall Correlation: {real_data['correlation_overall']:.3f} (p={real_data['p_value_overall']:.3f})")
            print(f"Correlation by type: {real_data['correlation_by_type']:.3f} (p={real_data['p_value_by_type']:.3f})")

            print("\nArtificial stories:")
            art_data = learning_results['artificial']
            print(f"Overall Correlation: {art_data['correlation_overall']:.3f} (p={art_data['p_value_overall']:.3f})")
            print(f"Correlation by type: {art_data['correlation_by_type']:.3f} (p={art_data['p_value_by_type']:.3f})")

            print("\nOverall:")
            ov_data = learning_results['overall']
            print(f"Overall Correlation: {ov_data['correlation']:.3f} (p={ov_data['p_value']:.3f})")

    # =========================================================================
    # Participant self-reported experience
    # =========================================================================
    if not NO_HUMAN_DATA:
        print("\n" + "=" * 60)
        print("PARTICIPANT EXPERIENCE")
        print("=" * 60 + "\n")
        analyzer.print_participant_experience()

    # =========================================================================
    # Reading curve stats (FPUB + human FP per story, for LaTeX subcaptions)
    # =========================================================================
    if RUN_READING_CURVE_STATS and not NO_HUMAN_DATA:
        print("\n" + "=" * 60)
        print("READING CURVE STATS (FPUB + Human FP per story)")
        print("=" * 60 + "\n")
        analyzer.set_naive_reader_mode("default")
        analyzer.print_reading_curve_stats()

    # =========================================================================
    # Name diversity metric (entropy per role from _w_details2.json files)
    # =========================================================================
    if RUN_NAME_DIVERSITY:
        print("\n" + "=" * 60)
        print("NAME DIVERSITY")
        print("=" * 60 + "\n")
        analyzer.compute_name_diversity()

    # =========================================================================
    # Model ordering test
    # =========================================================================
    if RUN_MODEL_ORDERING_TEST:
        analyzer.analyze_model_ordering(naive_mode=PNAS_NAIVE_MODE)

    if RUN_MODEL_ORDERING_PLOT:
        analyzer.plot_model_ordering(
            save_path=os.path.join(_PLOTS_DIR, "model_ordering.png"),
        )

    if RUN_HASSE_EDGE_COLORED:
        analyzer.plot_hasse_edge_colored(
            naive_mode=PNAS_NAIVE_MODE,
            save_path=os.path.join(_PLOTS_DIR, "model_ordering_edge_colored.png"),
        )
        analyzer.plot_hasse_edge_colored(
            naive_mode=PNAS_NAIVE_MODE,
            save_path=os.path.join(_PLOTS_DIR, "model_ordering_edge_colored_h.png"),
            horizontal=True,
        )

    if RUN_MODEL_METRICS_HEATMAP:
        analyzer.plot_model_metrics_heatmap(
            save_path=os.path.join(_PLOTS_DIR, "model_metrics_heatmap.png"),
        )

    # =========================================================================
    # Revelation point comparison
    # =========================================================================
    if RUN_REVELATION_COMPARISON:
        analyzer.compare_revelation_points(stories_dir=STORIES_DIR)

    # =========================================================================
    # PNAS figures
    # =========================================================================
    if RUN_PNAS_FIGURES:
        print("\n" + "=" * 60)
        print("GENERATING PNAS FIGURES")
        print("=" * 60 + "\n")
        analyzer.plot_pnas_figures(
            naive_mode=PNAS_NAIVE_MODE,
            exp_setting=PNAS_EXP_SETTING,
            save_prefix=PNAS_SAVE_PREFIX,
            bump_exp_setting=ACTUAL_FAIRPLAY_EXP_SETTING,
        )
        analyzer.plot_bump_sc_fp(
            naive_mode=PNAS_NAIVE_MODE,
            save_prefix=PNAS_SAVE_PREFIX,
            include_human_fp1=BUMP_INCLUDE_HUMAN_FP1,
        )
        analyzer.plot_bump_fp_thresholds(
            naive_mode=PNAS_NAIVE_MODE,
            save_prefix=PNAS_SAVE_PREFIX,
            include_human_fp3=BUMP_INCLUDE_HUMAN_FP3,
        )