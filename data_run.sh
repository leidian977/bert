echo "下载预训练模型"
wget -nc https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
unzip -o chinese_L-12_H-768_A-12.zip

echo "添加环境变量"
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
export MY_DATASET=./dataset

echo "开始执行"
python run_classifier.py \
  --task_name=wsdm \
  --do_train=true \
  --do_eval=true \
  --do_predict=true \
  --data_dir=$MY_DATASET \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=128 \
  --train_batch_size=32 \
  --learning_rate=5e-5 \
  --num_train_epochs=1.0 \
  --output_dir=./dataset/output

echo "计算测试集准确度"
python cal_accuracy.py
