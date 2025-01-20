#!/bin/zsh
#SBATCH --gres=gpu:1,vmem:40G
#SBATCH --mem=40g
#SBATCH -c2
#SBATCH --exclude=cyril-01
#SBATCH --time=18:0:0
#SBATCH --mail-type=FAIL,TIME_LIMIT


# Enable verbose mode
set -x

cd Tokens2Words/src/

model="Llama-3.1-8B"
model_name="meta-llama/${model}"
output_dir="Tokens2Words/runs/vocab_expansion/${model}"

per_word_eval_max_samples=20

dataset="wiki40b"
data_split="test"
data_language="he"
max_length=512
calibration_dataset="wiki40b"
calibration_split="validation"

calibration_lr="0.0001"
calibration_num_epochs=1
calibration_num_samples=10000

# for patchscopes:
detokenization_decision_rule="1st_id_layer"
detokenization_decision_rule_E="1st_id_layer"
detokenization_max_valid_layer="18"

# for heuristic detokenization, without patchscopes:
#detokenization_layer="7"
#detokenization_layer_embedding="7"

translators_name="procrustes_v2_rms_all_layers_extract_x"

patchscopes_prompt="בעברית: X X X X"
patchscopes_cache="${output_dir}/patchscopes/${dataset}/${data_language}/prompt_beivrit_x_x_x_x.parquet"

exp_name="estimate_expansion_success/${dataset}/${data_language}/${data_split}/patchscopes_max_len_${max_length}_calibrate_${calibration_num_samples}samples_on_${calibration_dataset}_${calibration_split}_${calibration_num_epochs}epochs_lr${calibration_lr}_detok_layer_U_${detokenization_decision_rule}_E_${detokenization_decision_rule_E}"
#exp_name="estimate_expansion_success/${dataset}/${data_language}/${data_split}/max_len_${max_length}_calibrate_${calibration_num_samples}samples_on_${calibration_dataset}_${calibration_split}_${calibration_num_epochs}epochs_lr${calibration_lr}_detok_layer_U${detokenization_layer}_E${detokenization_layer_embedding}"
#exp_name="estimate_expansion_success/${dataset}/${data_language}/${data_split}/max_len_${max_length}_load_patchscopes_on500words_calibration_detok_layer_U${detokenization_layer}_E${detokenization_layer_embedding}"


extraction_prompt="X"

translators_path="${output_dir}/translators/v2/${translators_name}.pt"

calibrators_path="${output_dir}/${exp_name}/calibrators/"

# load calibrators trained elsewhere:
#calibrators_exp_name="estimate_expansion_success/${dataset}/${data_language}/${data_split}/patchscopes_max_len_512_calibrate_10000samples_on_wiki40b_validation_1epochs_lr0.0001_detok_layer_U_1st_id_layer_E_1st_id_layer"
#calibrators_path="${output_dir}/${calibrators_exp_name}/calibrators/"


words_list="ŒTokens2Words/word_lists/top_5k_hebrew_words_without_nikud.txt"


python -m tokens2words.run_new_vocab_success_estimate \
  --output_dir "${output_dir}" --words_list "${words_list}" \
  --translators_use_procrustes --translators_procrustes_normalize \
  --calibrate_new_entries --calibration_save_dir "${calibrators_path}" --calibration_lr "${calibration_lr}" \
  --calibration_dataset "${calibration_dataset}" --calibration_dataset_split "${calibration_split}" \
  --calibration_dataset_language "${data_language}" --calibration_num_epochs "${calibration_num_epochs}" \
  --translators_path "${translators_path}" --translators_use_procrustes --translators_procrustes_normalize \
  --eval_dataset "${dataset}" --exp_name "${exp_name}" \
  --extraction_prompt "${extraction_prompt}" \
  --model_name "${model_name}" --extraction_batch_size 32  \
  --eval_batch_size 4 --calibration_batch_size 4 --eval_dataset_split "${data_split}" \
  --eval_dataset_language "${data_language}" --eval_max_length "${max_length}" \
  --eval_max_samples "${per_word_eval_max_samples}" \
  --calibration_max_samples "${calibration_num_samples}" \
  --use_patchscopes --patchscopes_results_cache "${patchscopes_cache}" \
  --patchscopes_prompt "${patchscopes_prompt}" \
  --detokenization_max_valid_layer ${detokenization_max_valid_layer} \
  --detokenization_decision_rule "${detokenization_decision_rule}" \
  --detokenization_decision_rule_E "${detokenization_decision_rule_E}" \
#  --overwrite_calibration \



## without patchscopes:
#python -m tokens2words.run_new_vocab_success_estimate \
#  --output_dir "${output_dir}" --words_list "${words_list}" --max_words 500 \
#  --translators_use_procrustes --translators_procrustes_normalize \
#  --calibrate_new_entries --calibration_save_dir "${calibrators_path}" --calibration_lr "${calibration_lr}" \
#  --calibration_dataset "${calibration_dataset}" --calibration_dataset_split "${calibration_split}" \
#  --calibration_dataset_language "${data_language}" --calibration_num_epochs "${calibration_num_epochs}" \
#  --translators_path "${translators_path}" --translators_use_procrustes --translators_procrustes_normalize \
#  --eval_dataset "${dataset}" --exp_name "${exp_name}" \
#  --extraction_prompt "${extraction_prompt}" \
#  --model_name "${model_name}" --extraction_batch_size 32  \
#  --eval_batch_size 4 --calibration_batch_size 4 --eval_dataset_split "${data_split}" \
#  --eval_dataset_language "${data_language}" --eval_max_length "${max_length}" \
#  --eval_max_samples "${per_word_eval_max_samples}" \
#  --calibration_max_samples "${calibration_num_samples}" \
#  --detokenization_layer "${detokenization_layer}" \
#  --detokenization_layer_embedding "${detokenization_layer_embedding}"
##  --overwrite_calibration \


