#单个model
#intervl2
python /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/internvl2.py \
        --image_folder /home/wuhaihong/xinhai/backend/src/static/OCRBench_Images \
        --OCRBench_file /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/OCRBench.json \
        --save_name Intervl2 \
        --num_workers 1 \
        --model_url http://localhost:40005/v1/chat/completions \
        --output_folder /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/model_answer

#MiniCPM-Llama3-V-2_5
python /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/single_model.py \
        --image_folder /home/wuhaihong/xinhai/backend/src/static/OCRBench_Images \
        --OCRBench_file /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/OCRBench.json \
        --save_name MiniCPM-Llama3-V-2_5 \
        --num_workers 1 \
        --model_url http://localhost:40005/v1/chat/completions \
        --output_folder /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/model_answer


#llava
python /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/llava.py \
        --image_folder /home/wuhaihong/xinhai/backend/src/static/OCRBench_Images \
        --OCRBench_file /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/OCRBench.json \
        --save_name llava \
        --num_workers 1 \
        --model_url http://localhost:40005/v1/chat/completions \
        --output_folder /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/model_answer


python /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/internvl2.py \
        --image_folder /home/wuhaihong/xinhai/backend/src/static/OCRBench_Images \
        --OCRBench_file /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/OCRBench.json \
        --save_name Intervl2 \
        --num_workers 1 \
        --model_url http://localhost:40005/v1/chat/completions \
        --output_folder /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/baseline

python /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/internvl2.py \
        --image_folder /home/wuhaihong/xinhai/backend/src/static/OCRBench_Images \
        --OCRBench_file /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/OCRBench.json \
        --save_name Intervl2 \
        --num_workers 1 \
        --model_url http://localhost:40005/v1/chat/completions \
        --output_folder /home/wuhaihong/xinhai/examples/OCRAgency/evaluation/ocr_benchmark/baseline
