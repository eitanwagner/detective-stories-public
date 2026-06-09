
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_path', default=os.path.dirname(os.path.abspath(__file__)) + "/", type=str, help="path to project root (with trailing slash)")
    parser.add_argument("-mt", "--model_type", type=str, default="openai", help="opeani, anthropic, gemini, llama")
    parser.add_argument("-mt2", "--model_type2", type=str, default="", help="The 'teacher' model. opeani, anthropic, gemini, llama")
    parser.add_argument("-mn", "--model_name", type=str, default="gpt-4o")
    parser.add_argument("-mn2", "--model_name2", type=str, default="")
    parser.add_argument("-st", "--story_type", type=str, default="", help="poirot, sherlock, rivals, gpt-4o, gpt-4o-mini, gemini-1.5-flash, gemini-1.5-pro")
    parser.add_argument('-s', '--surprising', action="store_true")

    parser.add_argument('-md', '--make_dict', action="store_true", help="whether to make a story dictionary")
    parser.add_argument('-ld', '--load_from_dict', action="store_true", help="whether to load a story dictionary")
    parser.add_argument('-ps', '--predict_strategies', action="store_true", help="whether to use all strategies for prediction")
    parser.add_argument('-es', '--estimation_strategies', action="store_true", help="whether to use all strategies for estimation")
    parser.add_argument('-es2', '--estimation_strategies2', action="store_true", help="whether to use only the best guessing strategy")
    parser.add_argument('-es3', '--estimation_strategies3', action="store_true", help="whether to use only the best guessing strategy")
    parser.add_argument('-ns', '--no_sampling', action="store_true", help="whether to skip sampling")
    parser.add_argument('-ls', '--less_sampling', action="store_true", help="whether to use less sampling points")
    parser.add_argument('-tb', '--thinking_budget', type=int, default=-1)

    parser.add_argument('-t', '--test', action="store_true")

    parser.add_argument("--id", type=int, default=0)
    parser.add_argument('-mr', '--make_revisions', action="store_true", help="whether to make revisions")
    parser.add_argument("-r", "--revisions", type=int, default=0)

    parser.add_argument('-sh', '--s_hint', action="store_true")
    parser.add_argument('-pa', '--plan_assumptions', action="store_true")
    parser.add_argument('-po', '--plan_outline', action="store_true")
    parser.add_argument('-pa2', '--plan_assumptions2', action="store_true", help="whether to use the assumption when performing sampling")
    parser.add_argument('-ch', '--c_hint', action="store_true")
    parser.add_argument('-rj', '--r_json', action="store_true")
    parser.add_argument('-c', '--clues', action="store_true")
    parser.add_argument('-tc', '--test_clues', action="store_true")
    parser.add_argument('-tct', '--test_clues_type', type=str, default="")

    parser.add_argument("-np", "--num_paragraphs", type=int, default=20, help="number of paragraphs for generation or prediction")
    # argument for number of samples
    parser.add_argument("-nsamp", "--num_samples", type=int, default=5, help="number of samples for sampling-based prediction")

    args = parser.parse_args()
    return args
