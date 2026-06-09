"""Core DetectiveStoryAnalyzer class: data loading and distribution computation."""

import pandas as pd
import numpy as np
import json
from pathlib import Path

from config import (
    model_dict, r_model_dict, NORMALIZE_NAIVE, NAIVE_THRESHOLD,
    USE_ARGMAX, REAL_STORY_S_MODEL, ignore, EXPERIENCED_READER_LABELS,
    EXPERIENCED_READER_SHORT_LABELS, EXPERIENCED_READER_COLORS
)
from utils import clopper_pearson_interval, argmax_distribution


class DetectiveStoryAnalyzer:
    def __init__(self, excel_path, json_dir, model_dict):
        self.excel_path = excel_path
        self.json_dir = Path(json_dir)
        self.model_dict = model_dict
        self.human_data = {}
        self.llm_data = {}
        self.experienced_reader_data = {}      # from -es2 files (settings 0, 1, 2)
        self.experienced_reader_data_es3 = {}  # from -es3 files (settings 3→0, 4→1, 5→2 internally)
        self.naive_reader_mode = None
        self.story_details = {}  # (model_key, story_id_str) -> story dict, loaded lazily

    # =========================================================================
    # Suspect list helper
    # =========================================================================

    def get_suspect_list(self, story_name, story_id):
        for k, v in self.llm_data.items():
            if (story_name == k[0] or story_name == k[1]) and story_id == k[2]:
                return v['data']['suspects']

    # =========================================================================
    # Human data loading
    # =========================================================================

    def load_human_data(self):
        """Load human responses from Excel file."""
        xl_file = pd.ExcelFile(self.excel_path)

        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(xl_file, sheet_name=sheet_name)

            if len(df.columns) <= 0 or len(df) == 0:
                continue

            form_col = df.columns[2]
            story_name = ""
            if "Real" in form_col:
                story_name = form_col[form_col.find("Real"):form_col.find("Real") + len("Real_0 0")]
            elif "Model" in form_col:
                story_name = form_col[form_col.find("Model"):form_col.find("Model") + len("Model_0 0")]
            story_id = story_name[-1]

            story_name = r_model_dict[story_name.split()[0]]

            suspects = self.get_suspect_list(story_name, story_id)

            response_cols = []
            for col in df.columns[3:]:
                if 'familiarity' in col.lower() or 'effort' in col.lower() or \
                        'surprising' in col.lower() or 'fair' in col.lower() or \
                        'coherent' in col.lower() or 'enjoyable' in col.lower():
                    break
                response_cols.append(col)

            self.human_data[story_name + " " + story_id] = {
                'df': df,
                'response_cols': response_cols,
                'ratings': df[df.columns[-6:]],
                'timestamps': df[df.columns[0]],
                'user_ids': df[df.columns[1]],
                'suspects': suspects
            }

        # Filter out stories that have no corresponding LLM sampling results
        self._filter_human_data_by_llm_availability()

    def _filter_human_data_by_llm_availability(self):
        """Remove human data entries for stories that have no corresponding LLM
        sampling results (e.g., because the JSON result files were removed).

        This ensures all downstream analyses only consider stories with both
        human and LLM data available.
        """
        stories_to_remove = []

        for story_name in list(self.human_data.keys()):
            model_key, story_id = story_name.split()
            is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

            # Check if LLM data exists for this story
            if not is_artificial:
                # Real stories use REAL_STORY_S_MODEL
                key = (REAL_STORY_S_MODEL, model_key, story_id)
            else:
                # Artificial stories: model_key is both the model and story type
                model_key2 = model_key if "gemini-2.5-flash" not in model_key else "gemini-2.5-flash"
                key = (model_key, model_key2, story_id)

            if key not in self.llm_data:
                stories_to_remove.append(story_name)

        if stories_to_remove:
            print(f"\nFiltering human data: removing {len(stories_to_remove)} stories without LLM sampling results:")
            for s in stories_to_remove:
                print(f"  - {s}")
                del self.human_data[s]
            print(f"Remaining human stories: {len(self.human_data)}\n")
        else:
            print(f"\nAll {len(self.human_data)} human stories have corresponding LLM data.\n")

    # =========================================================================
    # LLM data loading
    # =========================================================================

    def load_llm_data(self):
        """Load LLM experiment results from JSON files."""
        for json_file in self.json_dir.glob("*.json"):
            filename = json_file.stem

            parts = filename.split()
            params = {}
            i = 0
            while i < len(parts):
                if parts[i].startswith('-') and parts[i] != "-1":
                    key = parts[i].lstrip('-')
                    if i + 1 < len(parts) and (not parts[i + 1].startswith('-') or parts[i + 1] == "-1"):
                        params[key] = parts[i + 1]
                        i += 2
                    else:
                        params[key] = True
                        i += 1
                else:
                    i += 1

            model_name = params.get('mn')
            if model_name == 'gemini-2.5-flash' and 'tb' in params:
                tb_value = params['tb']
                model_name = f"gemini-2.5-flash-tb{tb_value}"
                params['mn'] = model_name

            with open(json_file, 'r') as f:
                data = json.load(f)

            key = (params.get('mn'), params.get('st'), params.get('id', '-1'))
            self.llm_data[key] = {
                'params': params,
                'data': data,
                'filename': filename
            }

    # =========================================================================
    # Experienced reader data loading (NEW)
    # =========================================================================

    def load_experienced_reader_data(self, with_prev_dir=None):
        """Load experienced reader results from JSON files with -es2 and -es3 flags.

        These files are in the with_prev/ subdirectory.

        -es2 files contain three 'direct' settings (stored as settings 0, 1, 2):
            direct["0"] — predict culprit (general instructions only, 0-shot)
            direct["1"] — 9 previous stories from same model given
            direct["2"] — 3 previous stories from same model given

        -es3 files contain three 'direct' settings (stored internally as 0, 1, 2,
        but exposed as settings 3, 4, 5 in the public API):
            direct["0"] — different prompt, no previous stories (0-shot alt)
            direct["1"] — 2 previous stories from same model given
            direct["2"] — 5 previous stories from same model given

        Args:
            with_prev_dir: Path to the with_prev directory. If None, defaults
                           to json_dir / "with_prev".
        """
        if with_prev_dir is None:
            with_prev_dir = self.json_dir / "with_prev"
        with_prev_dir = Path(with_prev_dir)

        if not with_prev_dir.exists():
            print(f"Warning: with_prev directory not found at {with_prev_dir}")
            return

        count_es2 = 0
        count_es3 = 0

        for json_file in with_prev_dir.glob("*.json"):
            filename = json_file.stem

            is_es2 = "-es2" in filename
            is_es3 = "-es3" in filename

            if not is_es2 and not is_es3:
                continue

            parts = filename.split()
            params = {}
            i = 0
            while i < len(parts):
                if parts[i].startswith('-') and parts[i] != "-1":
                    key = parts[i].lstrip('-')
                    if i + 1 < len(parts) and (not parts[i + 1].startswith('-') or parts[i + 1] == "-1"):
                        params[key] = parts[i + 1]
                        i += 2
                    else:
                        params[key] = True
                        i += 1
                else:
                    i += 1

            model_name = params.get('mn')
            if model_name == 'gemini-2.5-flash' and 'tb' in params:
                tb_value = params['tb']
                model_name = f"gemini-2.5-flash-tb{tb_value}"
                params['mn'] = model_name

            with open(json_file, 'r') as f:
                data = json.load(f)

            key = (params.get('mn'), params.get('st'), params.get('id', '-1'))

            entry = {
                'params': params,
                'data': data,
                'filename': filename
            }

            if is_es2:
                self.experienced_reader_data[key] = entry
                count_es2 += 1
            else:  # is_es3
                self.experienced_reader_data_es3[key] = entry
                count_es3 += 1

        print(f"Loaded {count_es2} experienced reader (-es2) files from {with_prev_dir}")
        print(f"Loaded {count_es3} experienced reader (-es3) files from {with_prev_dir}")

    # =========================================================================
    # Naive reader mode
    # =========================================================================

    def set_naive_reader_mode(self, mode):
        """Set the mode for computing naive reader probabilities

        Args:
            mode: one of
                "default"               — LLM gullible reader as-is
                "no_distractor"         — gullible reader with distractor zeroed out
                "clueless"              — uniform (0.25) until revelation, then 1 for culprit
                "clueless_no_distractor"  — clueless with distractor zeroed
                "super_naive_threshold"   — 0 for true culprit until threshold-based revelation,
                                            then 1; uniform over others before
                "super_naive_report"      — same step function, but revelation taken from
                                            the LLM-annotated true_culprit_paragraph field
        """
        valid_modes = ["default", "no_distractor", "clueless", "clueless_no_distractor",
                       "super_naive_threshold", "super_naive_report"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        self.naive_reader_mode = mode
        print(f"Naive reader mode set to: {mode}")

    # =========================================================================
    # Distribution computation
    # =========================================================================

    def get_human_distribution(self, story_name, use_argmax=None):
        """Calculate distribution of culprit guesses at each timestep.

        Args:
            story_name: Name of the story
            use_argmax: If True, convert to argmax. If None, use global USE_ARGMAX setting.
        """
        if story_name not in self.human_data:
            return None, None, None

        data = self.human_data[story_name]
        df = data['df']
        response_cols = data['response_cols']

        distributions = []
        confidence_intervals = []

        # Determine whether to use argmax
        apply_argmax = USE_ARGMAX if use_argmax is None else use_argmax

        for col in response_cols:
            responses = df[col].dropna()

            if len(responses) == 0:
                distributions.append([0, 0, 0, 0])
                confidence_intervals.append(([0, 0, 0, 0], [0, 0, 0, 0]))
                continue

            counts = [0, 0, 0, 0]
            for resp in responses:
                if resp not in data["suspects"]:
                    print("response not in suspect list!")
                    continue
                resp_idx = data["suspects"].index(resp) + 1
                if 1 <= resp_idx <= 4:
                    counts[resp_idx - 1] += 1

            total = sum(counts)
            probs = [c / total if total > 0 else 0 for c in counts]

            # Use Clopper-Pearson confidence intervals BEFORE argmax
            cp_lower = []
            cp_upper = []
            for count in counts:
                lower, upper = clopper_pearson_interval(count, total)
                cp_lower.append(lower)
                cp_upper.append(upper)
            confidence_intervals.append((cp_lower, cp_upper))

            if apply_argmax:
                probs_arr = np.array(probs)
                max_val = np.max(probs_arr)
                max_mask = (probs_arr == max_val)
                probs = (max_mask / max_mask.sum()).tolist()

            distributions.append(probs)

        return np.array(distributions), confidence_intervals, len(df)

    def get_llm_distributions(self, model_name, story_type, story_id, use_argmax=None):
        """Get know-it-all and naive reader distributions for a story.

        Args:
            model_name: Name of the model
            story_type: Type of story
            story_id: ID of the story
            use_argmax: If True, convert to argmax. If None, use global USE_ARGMAX setting.
        """
        key = (model_name, story_type, story_id)

        if key not in self.llm_data:
            return None, None, None, None, None

        data = self.llm_data[key]['data']
        params = self.llm_data[key]['params']
        nsamp = int(params.get('nsamp', '5'))  # Get number of samples

        ids = data.get('ids', [])

        # Determine whether to use argmax
        apply_argmax = USE_ARGMAX if use_argmax is None else use_argmax

        direct = data.get('direct', [])
        if len(direct) <= 0:
            naive_dist = None
            naive_ci = None
        else:
            for i, d in enumerate(direct["1"]):
                if len(d) < 4:
                    print(f"Made correction: {story_type} {story_id}")
                    direct["1"][i] = [0.25, 0.25, 0.25, 0.25]
            naive_dist = np.array(direct["1"])
            naive_dist = naive_dist / (naive_dist.sum(axis=1, keepdims=True) + 1e-6)
            # For naive (direct), we assume it's from a single run, so no CI
            naive_ci = None

        sampling_dist = np.array(data.get('sampling', []))

        # Calculate Clopper-Pearson confidence intervals for sampling distribution BEFORE argmax
        sampling_ci = []
        for timestep in sampling_dist:
            cp_lower = []
            cp_upper = []
            for prob in timestep:
                # Convert probability back to counts
                count = int(round(prob * nsamp))
                lower, upper = clopper_pearson_interval(count, nsamp)
                cp_lower.append(lower)
                cp_upper.append(upper)
            sampling_ci.append((cp_lower, cp_upper))

        suspects = data.get('suspects', [])
        real_probs = data.get('real_probs', [])

        if real_probs:
            true_culprit_idx = np.argmax(real_probs)
            if np.argmax(sampling_dist[-1]) != true_culprit_idx or max(sampling_dist[-1]) < 0.7:
                print(f"Discrepancy in true culprit index for {story_type} {story_id}")
                true_culprit_idx = None
        elif naive_dist is not None and len(naive_dist) > 0:
            true_culprit_idx = np.argmax(naive_dist[-1])
            if np.argmax(sampling_dist[-1]) != true_culprit_idx:
                print(f"Discrepancy in true culprit index for {story_type} {story_id}")
                true_culprit_idx = None
        else:
            true_culprit_idx = 0

        if NORMALIZE_NAIVE:
            distractor_probs = data.get('distractor_probs', [])
            if distractor_probs:
                dis_culprit_idx = np.argmax(distractor_probs)
                naive_dist[:, dis_culprit_idx] = 0.
                naive_dist = naive_dist / (naive_dist.sum(axis=1, keepdims=True) + 1e-6)

        if apply_argmax:
            sampling_dist = argmax_distribution(sampling_dist)
            if naive_dist is not None:
                naive_dist = argmax_distribution(naive_dist)

        return sampling_dist, naive_dist, true_culprit_idx, sampling_ci, nsamp

    # =========================================================================
    # Experienced reader distributions (NEW)
    # =========================================================================

    def get_experienced_reader_distributions(self, model_name, story_type, story_id,
                                              setting=2, use_argmax=None):
        """Get experienced reader distributions for a story.

        Settings 0-2 come from -es2 files:
            0 = predict culprit (general instructions only, 0-shot)
            1 = 9 previous stories given
            2 = 3 previous stories given

        Settings 3-5 come from -es3 files (internally stored as keys "0", "1", "2"):
            3 = different prompt, no previous stories (0-shot alt)
            4 = 2 previous stories given
            5 = 5 previous stories given

        Args:
            model_name: Name of the model (mn parameter)
            story_type: Type of story (st parameter)
            story_id: ID of the story
            setting: Which direct setting to use (0-5; see above).
            use_argmax: If True, convert to argmax. If None, use global USE_ARGMAX.

        Returns:
            experienced_dist: np.ndarray or None — distribution over suspects per timestep
            true_idx: int or None — index of true culprit
        """
        key = (model_name, story_type, story_id)
        apply_argmax = USE_ARGMAX if use_argmax is None else use_argmax

        # Determine which data store and which internal key to use
        if setting <= 2:
            data_store = self.experienced_reader_data
            internal_key = str(setting)
        else:
            data_store = self.experienced_reader_data_es3
            internal_key = str(setting - 3)  # 3→"0", 4→"1", 5→"2"

        if key not in data_store:
            return None, None

        data = data_store[key]['data']
        direct = data.get('direct', {})

        if internal_key not in direct or len(direct[internal_key]) == 0:
            return None, None

        # Normalise
        for i, d in enumerate(direct[internal_key]):
            if len(d) < 4:
                direct[internal_key][i] = [0.25, 0.25, 0.25, 0.25]

        exp_dist = np.array(direct[internal_key])
        exp_dist = exp_dist / (exp_dist.sum(axis=1, keepdims=True) + 1e-6)

        # Determine true culprit from the same file
        real_probs = data.get('real_probs', [])
        sampling_dist = np.array(data.get('sampling', []))

        if real_probs and any(p > 0 for p in real_probs):
            true_idx = int(np.argmax(real_probs))
        elif len(sampling_dist) > 0:
            true_idx = int(np.argmax(sampling_dist[-1]))
        else:
            return None, None  # cannot determine true culprit without circular inference

        if apply_argmax:
            exp_dist = argmax_distribution(exp_dist)

        return exp_dist, true_idx

    # =========================================================================
    # Naive reader probabilities
    # =========================================================================

    def get_naive_reader_probs(self, story_name, use_argmax=None):
        """Get naive reader probabilities based on current mode

        Args:
            story_name: Name of the story
            use_argmax: If True, convert to argmax. If False, use raw distribution.
                        If None, use global USE_ARGMAX setting.
        """
        model_key, story_id = story_name.split()
        is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

        if not is_artificial:
            llm_key = (REAL_STORY_S_MODEL, model_key, story_id)
            sampling_dist, naive_dist, true_idx, sampling_ci, nsamp = self.get_llm_distributions(
                REAL_STORY_S_MODEL, model_key, story_id, use_argmax=use_argmax
            )
        else:
            model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
            llm_key = (model_key, model_key2, story_id)
            sampling_dist, naive_dist, true_idx, sampling_ci, nsamp = self.get_llm_distributions(
                model_key, model_key2, story_id, use_argmax=use_argmax
            )

        # ids: paragraph numbers (1-indexed) at which sampling was taken
        ids = self.llm_data.get(llm_key, {}).get('data', {}).get('ids', None)

        if naive_dist is None or true_idx is None:
            return None, None

        # Determine whether to apply argmax
        apply_argmax = USE_ARGMAX if use_argmax is None else use_argmax

        if self.naive_reader_mode == "default":
            # Already argmaxed inside get_llm_distributions if apply_argmax is set
            return naive_dist, true_idx

        elif self.naive_reader_mode in ["no_distractor", "clueless_no_distractor"]:
            # Remove distractor
            key = (model_key, model_key if is_artificial else model_key, story_id)
            if key in self.llm_data:
                distractor_probs = np.array(self.llm_data[key]['data'].get('distractor_probs', []))
                if len(distractor_probs) > 0:
                    dis_idx = np.argmax(distractor_probs)
                    naive_dist_copy = naive_dist.copy()
                    naive_dist_copy[:, dis_idx] = 0.
                    naive_dist_copy = naive_dist_copy / (naive_dist_copy.sum(axis=1, keepdims=True) + 1e-6)

                    if self.naive_reader_mode == "clueless_no_distractor":
                        result, idx = self._apply_clueless_logic(naive_dist_copy, true_idx, dis_idx=dis_idx)
                        result_copy = result.copy()
                        result_copy[:, dis_idx] = 0.
                        result = result_copy / (result_copy.sum(axis=1, keepdims=True) + 1e-6)
                        if apply_argmax:
                            result = argmax_distribution(result)
                        return result, idx
                    if apply_argmax:
                        naive_dist_copy = argmax_distribution(naive_dist_copy)
                    return naive_dist_copy, true_idx

            if self.naive_reader_mode == "clueless_no_distractor":
                result, idx = self._apply_clueless_logic(naive_dist, true_idx)
                if apply_argmax:
                    result = argmax_distribution(result)
                return result, idx
            # no_distractor but no distractor_probs found — fall through unchanged
            return naive_dist, true_idx

        elif self.naive_reader_mode == "clueless":
            result, idx = self._apply_clueless_logic(naive_dist, true_idx)
            if apply_argmax:
                result = argmax_distribution(result)
            return result, idx

        elif self.naive_reader_mode == "super_naive_threshold":
            result, idx = self._apply_super_naive_logic(naive_dist, true_idx)
            if apply_argmax:
                result = argmax_distribution(result)
            return result, idx

        elif self.naive_reader_mode == "super_naive_report":
            result, idx = self._apply_super_naive_report_logic(naive_dist, true_idx, model_key, story_id, ids=ids)
            if apply_argmax:
                result = argmax_distribution(result)
            return result, idx

    def _apply_clueless_logic(self, naive_dist, true_idx, dis_idx=None):
        """Apply clueless logic to a naive distribution"""
        true_probs = naive_dist[:, true_idx]
        revelation_idx = len(naive_dist) - 1

        for i in range(len(true_probs)):
            if true_probs[i] >= NAIVE_THRESHOLD:
                if all(true_probs[j] >= NAIVE_THRESHOLD for j in range(i, len(true_probs))):
                    revelation_idx = i
                    break

        clueless_dist = np.ones_like(naive_dist) * 0.25
        clueless_dist[revelation_idx:, true_idx] = 1.0
        for j in range(4):
            if j != true_idx:
                clueless_dist[revelation_idx:, j] = 0.0

        if dis_idx is not None:
            naive_dist_copy = naive_dist.copy()
            naive_dist_copy[:, dis_idx] = 0.
            clueless_dist = naive_dist_copy / (naive_dist_copy.sum(axis=1, keepdims=True) + 1e-6)

        return clueless_dist, true_idx

    def _apply_super_naive_logic(self, naive_dist, true_idx):
        """Super-naive (threshold): 0 for true culprit until threshold-based revelation, then 1.

        Revelation is the first step where naive prob >= NAIVE_THRESHOLD and stays above it.
        Before revelation: uniform over all non-true suspects.
        """
        n_steps, n_suspects = naive_dist.shape
        true_probs = naive_dist[:, true_idx]
        revelation_idx = n_steps - 1

        for i in range(len(true_probs)):
            if true_probs[i] >= NAIVE_THRESHOLD:
                if all(true_probs[j] >= NAIVE_THRESHOLD for j in range(i, len(true_probs))):
                    revelation_idx = i
                    break

        return self._build_super_naive_dist(n_steps, n_suspects, true_idx, revelation_idx)

    def _apply_super_naive_report_logic(self, naive_dist, true_idx, model_key, story_id, ids=None):
        """Super-naive (report): same step function but revelation from LLM-annotated paragraph.

        Uses the true_culprit_paragraph field from the _w_details*.json story files.
        Falls back to the last step if the field is missing.

        Args:
            ids: list of 1-indexed paragraph numbers at which sampling was taken.
                 Used to convert true_culprit_paragraph → waypoint index correctly.
                 If None, falls back to treating para as a direct step index.
        """
        self._load_story_details_if_needed()

        n_steps, n_suspects = naive_dist.shape
        revelation_idx = n_steps - 1

        story = self.story_details.get((model_key, str(story_id)))
        if story is not None:
            para = story.get('true_culprit_paragraph')
            if para is not None:
                para = int(para)
                if ids is not None:
                    # Find the first waypoint that covers the revelation paragraph
                    matching = [i for i, p in enumerate(ids) if p >= para]
                    revelation_idx = matching[0] if matching else n_steps - 1
                else:
                    # Fallback: treat paragraph number as 1-indexed step
                    revelation_idx = max(0, min(para - 1, n_steps - 1))

        return self._build_super_naive_dist(n_steps, n_suspects, true_idx, revelation_idx)

    def _build_super_naive_dist(self, n_steps, n_suspects, true_idx, revelation_idx):
        """Build the step-function distribution shared by both super-naive modes."""
        n_others = n_suspects - 1
        dist = np.zeros((n_steps, n_suspects))
        for j in range(n_suspects):
            if j != true_idx:
                dist[:revelation_idx, j] = 1.0 / n_others
        dist[revelation_idx:, true_idx] = 1.0
        return dist, true_idx

    def _load_story_details_if_needed(self, stories_dir=None):
        """Lazily load _w_details*.json story files into self.story_details.

        Keyed by (model_key, story_id_str) where story_id_str is the 0-based list index.
        Prefers _w_details2.json over _w_details.json when both exist.
        """
        if self.story_details:
            return

        if stories_dir is None:
            stories_dir = self.json_dir.parent.parent / "stories"
        stories_dir = Path(stories_dir)

        # Collect files, preferring _w_details2 over _w_details for each stem base.
        # Skip _po_ and _pa_ variants (stories written to match real-story plots).
        best = {}  # base_stem -> Path
        for f in stories_dir.glob('*_w_details*.json'):
            stem = f.stem
            if '_po_' in stem or '_pa_' in stem:
                continue
            base = stem.replace('_w_details2', '').replace('_w_details', '')
            # _w_details2 wins over _w_details
            if base not in best or '_w_details2' in stem:
                best[base] = f

        for base, path in best.items():
            model_part = base.split('_w_suspects')[0]
            # gemini-2.5-flash uses tb as part of its name (e.g. gemini-2.5-flash-tb0);
            # all other models use tb only as a generation parameter → strip it.
            base_model = model_part.split('_tb')[0]
            if base_model == 'gemini-2.5-flash' and '_tb' in model_part:
                model_key = model_part.replace('_tb', '-tb')
            else:
                model_key = base_model

            try:
                with open(path) as f:
                    stories = json.load(f)
            except Exception:
                continue

            if isinstance(stories, dict):
                stories = list(stories.values())

            for i, story in enumerate(stories):
                self.story_details[(model_key, str(i))] = story

    # =========================================================================
    # Helper: resolve LLM key for a human story
    # =========================================================================

    def _resolve_llm_key(self, story_name):
        """Given a human story name like 'gemini-2.5-pro 3', return the
        (model_name, story_type, story_id) key for llm_data lookups and
        whether the story is artificial."""
        model_key, story_id = story_name.split()
        is_artificial = not (model_key.startswith('poirot') or model_key.startswith("sherlock"))

        if not is_artificial:
            key = (REAL_STORY_S_MODEL, model_key, story_id)
        else:
            model_key2 = model_key if "gemini-2.5-flash" not in story_name else "gemini-2.5-flash"
            key = (model_key, model_key2, story_id)

        return key, is_artificial