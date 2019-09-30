BERT_BASE_DIR="你的bert模型路径"
DATA_DIR=../process_data
OUTPUT_DIR=../model

python data_process.py;
python BERT-BiLSTM-CRF-NER/run.py \
    -data_dir=${DATA_DIR} \
    -output_dir=${OUTPUT_DIR} \
    -init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
    -bert_config_file=${BERT_BASE_DIR}/bert_config.json \
    -vocab_file=${BERT_BASE_DIR}/vocab.txt \
    -batch_size=4 \
    -max_seq_length=512 \
    -learning_rate=5e-6 \
    -save_checkpoint_max=1 \
    -save_summary_steps=5000 \
    -save_checkpoints_steps=2500 \
    -device_map="0"/
python post_process.py;

