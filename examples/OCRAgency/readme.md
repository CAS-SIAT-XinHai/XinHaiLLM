## OCRagency v1:

​	agent1:OCR+llm ：先ocr识别图片文本，然后再llm整理出符合回答要求模板的答案

​	agent2:mllm：直接让多模态回答

​	agent0:验证llm：验证前面两个agent回答，形成最终答案。



## Quick start

​	1.git clone https://github.com/wurevvc/XinHaiLLM.git

​	2.cd XinHaiLLM

​	3.按照examples/OCRagency里面的start_example.sh的配置来配置相应的服务。

​	4.启动simulation来跑结果,配置文件为xinhai_ocr.yaml

​	5.simulation文件中的run函数中的user_question则为用户问题，image_path为图片路径，save_path为保存的地址

​	

​	6.配置文件中重要的参数:
