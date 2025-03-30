#!/bin/zsh


cd Tokens2Words/src/

model="Llama-2-7b-hf"
model_name="meta-llama/${model}"
output_dir="Tokens2Words/runs/vocab_expansion/${model}"
output_dir="/cs/snapless/roys/yuval.reif/Tokens2Words/runs/vocab_expansion/debug/${model}/"

#dataset="wikitext"
#data_split="test"
#data_language="en"
#max_length=512
#min_freq="1"
#calibration_dataset="wikitext"
#calibration_split="train"

dataset="pubmed"
data_split="test"
data_language="en"
max_length=512
min_freq="5"
calibration_dataset="pubmed"
calibration_split="train"

#dataset="wiki40b"
#data_split="test"
#data_language="fr"
#max_length=512
#min_freq="50"
#calibration_dataset="wiki40b"
#calibration_split="validation"


calibration_lr="0.0001"
calibration_num_epochs=1

detokenization_decision_rule="1st_id_layer"
detokenization_decision_rule_E="1st_id_layer"

procrustes_layer="all"
translators_name="translators"
translators_path="${output_dir}/translators.pt"

#patchscopes_prompt="X, X, X, X,"
patchscopes_prompt="X X X X"
patchscopes_cache="${output_dir}/patchscopes/${dataset}/prompt_x_x_x_x.parquet"

extraction_prompt="X"


exp_name="${dataset}/${data_language}/full/${data_split}/max_len_${max_length}_calibration_E_and_U_on_${calibration_dataset}_${calibration_split}_${calibration_num_epochs}epochs_lr${calibration_lr}_${translators_name}_min_freq${min_freq}_patchscopes_x_x_x_x_${detokenization_decision_rule}_and_E_${detokenization_decision_rule_E}"

calibrators_path="${output_dir}/${exp_name}/calibrators/"



python -m tokens2words.run_vocab_expansion_eval \
    --output_dir "${output_dir}"  \
    --model_name "${model_name}" \
    --exp_name "${exp_name}" \
    --calibrate_new_entries --calibration_save_dir "${calibrators_path}" --calibration_lr "${calibration_lr}" \
    --calibration_dataset "${calibration_dataset}" --calibration_dataset_split "${calibration_split}" \
    --calibration_dataset_language "${data_language}" --calibration_max_samples 10000 \
    --calibration_num_epochs "${calibration_num_epochs}" \
    --words_dataset "${dataset}" --words_dataset_split "${data_split}" \
    --words_filter_non_en --words_filter_min_freq "${min_freq}" \
    --extraction_prompt "${extraction_prompt}" --patchscopes_prompt "${patchscopes_prompt}" \
    --detokenization_decision_rule "${detokenization_decision_rule}" \
    --detokenization_decision_rule_E "${detokenization_decision_rule_E}" \
    --extraction_batch_size 32  --eval_batch_size 4 --calibration_batch_size 4 \
    --eval_dataset "${dataset}" --eval_dataset_split "${data_split}" \
    --eval_max_length "${max_length}" --eval_max_samples 10000 \
    --translators_path "${translators_path}" \
    --patchscopes_results_cache "${patchscopes_cache}"
