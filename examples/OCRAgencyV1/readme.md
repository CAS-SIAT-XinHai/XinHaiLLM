## OCRagency v1:

​	agent1:OCR+llm ：先ocr识别图片文本，然后再llm整理出符合回答要求模板的答案

​	agent2:mllm：直接让多模态回答

​	agent0:验证llm：验证前面两个agent回答，形成最终答案。



## Quick start

​	1.用该文件夹下的backend替换原来的backend

​	2.按照配置文件的需求和接口，启动以下服务，mllm,llm,ocr

​	3.启动simulation，既可以进行模拟,配置文件为xinhai_ocr_V1.yaml

​	4.评估，eval文件夹下跑一次数据时候，都会对配置文件xinhai_ocr_V1.yaml进行修改，

​		主要修改的参数 environment/image_path:指定图片路径

​					environment/ user_question：用户问题

​					prompts/answer_template:  指定最终生成的模板
